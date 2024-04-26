###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional
from collections.abc import Mapping, Sequence


import numpy as np
import torch.distributed
import torch.nn.functional
from e3nn import o3
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage
from torch.utils.data.dataloader import default_collate



import mace
from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import (
    LRScheduler,
    create_error_table,
    get_atomic_energies,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
)
from mace.tools.slurm_distributed import DistributedEnvironment, SingleGPUEnvironment
from mace.tools.finetuning_utils import load_foundations, extract_config_mace_model

from mace.data import get_neighborhood, AtomicData
from mace.tools import (
    atomic_numbers_to_indices,
    to_one_hot,
    voigt_to_matrix,
    AtomicNumberTable,
)

from mace.tools.torch_geometric import Batch, Data


import sys
sys.path.insert(0, '/home/bepmusil/git/test/mlcg-tools/')
from mlcg.utils import load_yaml
from mlcg.pl import H5DataModule
from mlcg.data import AtomicData as MLCGData


def make_atomic_data(config, z_table, cutoff):
    positions = torch.as_tensor(config.pos, dtype=torch.get_default_dtype())

    pbc = config.get("pbc", None)
    if pbc is None:
        pbc = torch.zeros(3, dtype=torch.bool)
    else:
        pbc = torch.as_tensor(pbc, dtype=torch.bool)

    cell = config.get("cell", None)
    # print("$$$$", pbc, cell,torch.all(pbc), torch.all(pbc) == False)
    if cell is None and torch.all(pbc) == False:
        # move atoms to the positive region
        positions -= torch.min(positions,dim=0)[0]
        aa = positions.max(dim=0)[0]
        # deal with 2D configurations
        aa[aa == 0.] = 1.
        cell = torch.diag(aa)

    elif cell is None and torch.any(pbc) == True:
        raise RuntimeError("no cell specified but PBC active")
    else:
        torch.as_tensor(cell, dtype=torch.get_default_dtype())


    edge_index, shifts, unit_shifts = get_neighborhood(
        positions=positions.cpu().numpy(), cutoff=float(cutoff), pbc=pbc.numpy(), cell=cell.numpy()
    )
    indices = atomic_numbers_to_indices(config.atom_types , z_table=z_table)
    one_hot = to_one_hot(
        torch.as_tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )

    weight = (
        torch.as_tensor(config.weight, dtype=torch.get_default_dtype())
        if config.get("weight") is not None
        else None
    )

    energy_weight = (
        torch.as_tensor(config.energy_weight, dtype=torch.get_default_dtype())
        if config.get("energy_weight") is not None
        else None
    )

    forces_weight = (
        torch.as_tensor(config.forces_weight, dtype=torch.get_default_dtype())
        if config.get("forces_weight") is not None
        else None
    )

    stress_weight = (
        torch.as_tensor(config.stress_weight, dtype=torch.get_default_dtype())
        if config.get("stress_weight") is not None
        else None
    )

    virials_weight = (
        torch.as_tensor(config.virials_weight, dtype=torch.get_default_dtype())
        if config.get("virials_weight") is not None
        else None
    )

    forces = (
        torch.as_tensor(config.forces, dtype=torch.get_default_dtype())
        if config.get("forces") is not None
        else None
    )
    energy = (
        torch.as_tensor(config.energy, dtype=torch.get_default_dtype())
        if config.get("energy") is not None
        else None
    )
    stress = (
        voigt_to_matrix(
            torch.as_tensor(config.stress, dtype=torch.get_default_dtype())
        ).unsqueeze(0)
        if config.get("stress") is not None
        else None
    )
    virials = (
        voigt_to_matrix(
            torch.as_tensor(config.virials, dtype=torch.get_default_dtype())
        ).unsqueeze(0)
        if config.get("virials") is not None
        else None
    )
    dipole = (
        torch.as_tensor(config.dipole, dtype=torch.get_default_dtype()).unsqueeze(0)
        if config.get("dipole") is not None
        else None
    )
    charges = (
        torch.as_tensor(config.charges, dtype=torch.get_default_dtype())
        if config.get("charges") is not None
        else None
    )

    return AtomicData(
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        positions=positions,
        shifts=torch.as_tensor(shifts, dtype=torch.get_default_dtype()),
        unit_shifts=torch.as_tensor(unit_shifts, dtype=torch.get_default_dtype()),
        cell=cell,
        node_attrs=one_hot,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        forces=forces,
        energy=energy,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
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
            batch = [make_atomic_data(data, self.z_table, self.cutoff) for data in batch]
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
    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:  # pylint: disable=W0703
            logging.error(f"Failed to initialize distributed environment: {e}. falling back to single gpu environement")
            distr_env = SingleGPUEnvironment()
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        if rank == 0:
            print(distr_env)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = int(0)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        if rank == 0: logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        if rank == 0: logging.info(f"Processes: {world_size}")

    try:
        if rank == 0: logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        if rank == 0: logging.info("Cannot find MACE version, please install MACE via pip")
    if rank == 0: logging.info(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        if rank == 0: logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    h5_config = load_yaml(args.h5_config_fn)
    print("####################")
    print(h5_config)

    # Atomic number table
    # yapf: disable
    print("args.atom_types",args.atom_types)
    unique_atom_types = np.unique(args.atom_types)
    print("unique_atom_tags",unique_atom_types)
    z_table = tools.get_atomic_number_table_from_zs(
        unique_atom_types
    )

    # yapf: enable
    if rank == 0: logging.info(z_table)
    atomic_energies_dict = {z:0. for z in range(100)}
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    if rank == 0: logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    compute_energy = True
    args.compute_forces = True
    compute_virials = False
    args.compute_stress = False
    compute_dipole = False

    datamodule = H5DataModule(
        collater_fn=Collater(z_table, args.r_max),
        **h5_config
    )
    datamodule.prepare_data()
    datamodule.setup("stage")

    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)

    if rank == 0: logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        if args.distributed:
            num_graphs = torch.tensor(len(train_loader.dataset)).to(device)
            num_neighbors = num_graphs * torch.tensor(avg_num_neighbors).to(device)
            torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_neighbors, op=torch.distributed.ReduceOp.SUM
            )
            args.avg_num_neighbors = (num_neighbors / num_graphs).item()
        else:
            args.avg_num_neighbors = avg_num_neighbors
    if rank == 0: logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")


    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    if rank == 0: logging.info(f"Selected the following outputs: {output_args}")

    args.std = 1.0
    if rank == 0: logging.info("No scaling selected")


    # Build model
    if rank == 0: logging.info("Building model")
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
        cutoff_type=args.cutoff_type,
        radial_type=args.radial_type,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )

    model = modules.ScaleShiftMACE(
        **model_config,
        pair_repulsion=args.pair_repulsion,
        distance_transform=args.distance_transform,
        correlation=args.correlation,
        gate=modules.gate_dict[args.gate],
        activation=modules.gate_dict[args.activation],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        atomic_inter_scale=args.std,
        atomic_inter_shift=0.0,
        radial_MLP=ast.literal_eval(args.radial_MLP),
    )

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
        # assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        else:
            if args.start_swa > args.max_num_epochs:
                if rank == 0: logging.info(
                    f"Start swa must be less than max_num_epochs, got {args.start_swa} > {args.max_num_epochs}"
                )
                args.start_swa = args.max_num_epochs // 4 * 3
                logging.info(f"Setting start swa to {args.start_swa}")
        if args.loss == "forces_only":
            if rank == 0: logging.info("Can not select swa with forces only loss.")
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
            if rank == 0: logging.info(
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
        except Exception:  # pylint: disable=W0703
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
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if rank == 0: logging.info(model)
    if rank == 0: logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    if rank == 0: logging.info(f"Optimizer: {optimizer}")

    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

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
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        # train_sampler=train_sampler,
        rank=rank,
    )

    if rank == 0: logging.info("Computing metrics for training, validation, and test sets")

    all_data_loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }

    swa_eval = False
    epoch = checkpoint_handler.load_latest(
        state=tools.CheckpointState(model, optimizer, lr_scheduler),
        swa=swa_eval,
        device=device,
    )
    model.to(device)
    if rank == 0: logging.info(f"Loaded model from epoch {epoch}")

    for param in model.parameters():
        param.requires_grad = False
    table = create_error_table(
        table_type=args.error_table,
        all_data_loaders=all_data_loaders,
        model=model,
        loss_fn=loss_fn,
        output_args=output_args,
        log_wandb=args.wandb,
        device=device,
        distributed=args.distributed,
    )
    if rank == 0: logging.info("\n" + str(table))


    if rank == 0:
        # Save entire model
        model_path = Path(args.checkpoints_dir) / (tag + ".model")
        logging.info(f"Saving model to {model_path}")
        if args.save_cpu:
            model = model.to("cpu")
        torch.save(model, model_path)

        torch.save(model, Path(args.model_dir) / (args.name + ".model"))

    if args.distributed:
        torch.distributed.barrier()

    if rank == 0: logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")
    main()
