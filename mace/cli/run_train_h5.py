###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch.nn.functional
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.script_utils import (
    LRScheduler,
    create_error_table,
    create_error_table_h5,
    get_dataset_from_xyz,
)

import sys
sys.path.insert(0, '/home/bepmusil/git/mlcg-tools/')
from mlcg.utils import load_yaml
from mlcg.pl import H5DataModule
from mlcg.data import AtomicData as MLCGData
# from mlcg.datasets import H5Dataset, H5MetasetDataLoader


from typing import Dict, Mapping, Optional
import torch



from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from mace.tools.torch_geometric import Batch, Data

from mace.data import get_neighborhood, AtomicData
from mace.tools import (
    atomic_numbers_to_indices,
    to_one_hot,
)

def make_atomic_data(config, z_table, cutoff):
    pbc = config.get("pbc", None)
    cell = config.get("cell", None)
    edge_index, shifts = get_neighborhood(
        positions=config.pos.cpu().numpy(), cutoff=cutoff, pbc=pbc, cell=cell
    )
    indices = atomic_numbers_to_indices(config.atom_types , z_table=z_table)
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )

    cell = (
        torch.tensor(cell, dtype=torch.get_default_dtype())
        if cell is not None
        else None
    )

    weight = (
        torch.tensor(config.weight, dtype=torch.get_default_dtype())
        if config.get("weight") is not None
        else None
    )

    forces = (
        torch.tensor(config.forces, dtype=torch.get_default_dtype())
        if config.get("forces") is not None
        else None
    )
    energy = (
        torch.tensor(config.energy, dtype=torch.get_default_dtype())
        if config.get("energy") is not None
        else None
    )

    return AtomicData(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        positions=torch.tensor(config.pos, dtype=torch.get_default_dtype()),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        cell=cell,
        node_attrs=one_hot,
        weight=weight,
        forces=forces,
        energy=energy,
    )

class Collater:
    def __init__(self,z_table, cutoff, follow_batch=[], exclude_keys=[]):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.z_table = z_table
        self.cutoff = cutoff

    def __call__(self, batch):
        elem = batch[0]

        # convert to current MACE's AtomicData with NL
        if isinstance(elem, MLCGData):
            batch = [make_atomic_data(data,self.z_table,self.cutoff) for data in batch]
            elem = batch[0]

        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


def main() -> None:
    args = tools.build_h5_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    h5_config = load_yaml(args.h5_config_fn)
    print("####################")
    print(h5_config)


    # Data preparation
    # collections, atomic_energies_dict = get_dataset_from_xyz(
    #     train_path=args.train_file,
    #     valid_path=args.valid_file,
    #     valid_fraction=args.valid_fraction,
    #     config_type_weights=config_type_weights,
    #     test_path=args.test_file,
    #     seed=args.seed,
    #     energy_key=args.energy_key,
    #     forces_key=args.forces_key,
    #     stress_key=args.stress_key,
    #     virials_key=args.virials_key,
    #     dipole_key=args.dipole_key,
    #     charges_key=args.charges_key,
    # )

    # logging.info(
    #     f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
    #     f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    # )

    # Atomic number table
    # yapf: disable
    print("args.atom_types",args.atom_types)
    unique_atom_types = np.unique(args.atom_types)
    print("unique_atom_tags",unique_atom_types)
    z_table = tools.get_atomic_number_table_from_zs(
        unique_atom_types
    )
    #     z
    #     for configs in (collections.train, collections.valid)
    #     for config in configs
    #     for z in config.atomic_numbers
    # )
    # yapf: enable
    logging.info(z_table)
    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        args.compute_forces = False
        compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            args.compute_forces = True
            compute_virials = False
            args.compute_stress = False
        else:
            compute_energy = True
            args.compute_forces = True
            args.compute_stress = False
            compute_dipole = False
        atomic_energies_dict = {z:0. for z in range(100)}
        # if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        #     if args.E0s is not None:
        #         logging.info(
        #             "Atomic Energies not in training file, using command line argument E0s"
        #         )
        #         if args.E0s.lower() == "average":
        #             logging.info(
        #                 "Computing average Atomic Energies using least squares regression"
        #             )
        #             atomic_energies_dict = data.compute_average_E0s(
        #                 collections.train, z_table
        #             )
        #         else:
        #             try:
        #                 atomic_energies_dict = ast.literal_eval(args.E0s)
        #                 assert isinstance(atomic_energies_dict, dict)
        #             except Exception as e:
        #                 raise RuntimeError(
        #                     f"E0s specified invalidly, error {e} occured"
        #                 ) from e
        #     else:
        #         raise RuntimeError(
        #             "E0s not found in training file and not specified in command line"
        #         )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    datamodule = H5DataModule(
        collater_fn=Collater(z_table, args.r_max),
        **h5_config
    )
    datamodule.prepare_data()
    datamodule.setup("stage")

    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    # train_loader = torch_geometric.dataloader.DataLoader(
    #     dataset=[
    #         data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
    #         for config in collections.train
    #     ],
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True,
    # )
    # valid_loader = torch_geometric.dataloader.DataLoader(
    #     dataset=[
    #         data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
    #         for config in collections.valid
    #     ],
    #     batch_size=args.valid_batch_size,
    #     shuffle=False,
    #     drop_last=False,
    # )

    loss_fn: torch.nn.Module
    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    elif args.loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            virials_weight=args.virials_weight,
        )
    elif args.loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
        )
    elif args.loss == "huber":
        loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args.loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        # Unweighted Energy and Forces loss by default
        loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

    # Selecting outputs
    compute_virials = False
    if args.loss in ("stress", "virials", "huber"):
        compute_virials = True
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    logging.info("Building model")
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    logging.info(f"Hidden irreps: {args.hidden_irreps}")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )

    model: torch.nn.Module

    if args.model == "MACE":
        if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            activation=modules.gate_dict[args.activation],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "ScaleShiftMACE":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "ScaleShiftBOTNet":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "BOTNet":
        model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    elif args.model == "AtomicDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        model = modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            # dipole_scale=1,
            # dipole_shift=0,
        )
    elif args.model == "EnergyDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        model = modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    model.to(device)

    # Optimizer
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        if args.loss == "forces_only":
            logging.info("Can not select swa with forces only loss.")
        elif args.loss == "virials":
            loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                virials_weight=args.swa_virials_weight,
            )
        elif args.loss == "stress":
            loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                stress_weight=args.swa_stress_weight,
            )
        elif args.loss == "energy_forces_dipole":
            loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
                args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                dipole_weight=args.swa_dipole_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
            )
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=args.start_swa,
            loss_fn=loss_fn_energy,
        )

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception as e:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
        args_dict_json = json.dumps(args_dict)
        for key in args.wandb_log_hypers:
            wandb_config[key] = args_dict[key]
        tools.init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config,
        )
        wandb.run.summary["params"] = args_dict_json

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
    )

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    # all_collections = [
    #     ("train", collections.train),
    #     ("valid", collections.valid),
    # ] + collections.tests
    all_loaders = [
        ("train",train_loader),
        ("valid",valid_loader),
        ("test",test_loader),
    ]

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        logging.info(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False
        table = create_error_table_h5(
            table_type=args.error_table,
            all_loaders=all_loaders,
            z_table=z_table,
            r_max=args.r_max,
            valid_batch_size=args.valid_batch_size,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
        )
        logging.info("\n" + str(table))

        # Save entire model
        if swa_eval:
            model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
        else:
            model_path = Path(args.checkpoints_dir) / (tag + ".model")
        logging.info(f"Saving model to {model_path}")
        if args.save_cpu:
            model = model.to("cpu")
        torch.save(model, model_path)

        if swa_eval:
            torch.save(model, Path(args.model_dir) / (args.name + "_swa.model"))
        else:
            torch.save(model, Path(args.model_dir) / (args.name + ".model"))

    logging.info("Done")


if __name__ == "__main__":
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")

    main()
