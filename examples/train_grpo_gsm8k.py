"""Hydra configuration entry point for GRPO; training execution is not wired yet."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from kirl.config import register_configs, resolve_train_config

register_configs()


@hydra.main(
    version_base="1.3", config_path=str(Path(__file__).resolve().parents[1] / "configs"), config_name="train"
)
def main(cfg: DictConfig) -> None:
    config = resolve_train_config(
        cfg, base_dir=get_original_cwd(), output_dir=HydraConfig.get().runtime.output_dir
    )
    resolved = OmegaConf.create(config.resolved)
    snapshot = Path(config.resolved["output_dir"]) / "resolved_config.yaml"
    OmegaConf.save(resolved, snapshot, resolve=True)
    print(OmegaConf.to_yaml(resolved))
    print(f"Configuration validated and saved to {snapshot}. Training pipeline is not implemented yet.")


if __name__ == "__main__":
    main()
