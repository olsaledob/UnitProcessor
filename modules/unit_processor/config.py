from dataclasses import dataclass
from typing import Optional
import toml

@dataclass
class UnitProcessorConfig:
    # Data handling
    dt: float
    save_plots: bool
    max_per_row: int
    export_enabled: bool

    # Paths
    h5_dir: str
    led_dir: str
    export_dir: str
    plot_dir: str
    log_dir: str

    # Processing behaviour
    center: bool
    plotting: bool
    apply_filters: bool

    # Optional parameters for receptive-field analysis
    fraction_spikes: float
    seed: Optional[int]

    @classmethod
    def load(cls, toml_path: str) -> 'UnitProcessorConfig':
        cfg = toml.load(toml_path)

        up_cfg = cfg["unit_processor"]

        return cls(
            dt=float(up_cfg.get("dt", 0.01)),
            save_plots=bool(up_cfg.get("save_plots", False)),
            max_per_row=int(up_cfg.get("max_per_row", 5)),
            export_enabled=bool(up_cfg.get("export_enabled", True)),

            h5_dir=up_cfg["h5_dir"],
            led_dir=up_cfg["led_dir"],
            export_dir=up_cfg["export_dir"],
            plot_dir=up_cfg["plot_dir"],
            log_dir=up_cfg.get("log_dir", "./logs"),

            center=bool(up_cfg.get("center", True)),
            plotting=bool(up_cfg.get("plotting", True)),
            apply_filters=bool(up_cfg.get("apply_filters", False)),

            fraction_spikes=float(up_cfg.get("fraction_spikes", 1.0)),
            seed=(int(up_cfg["seed"]) if up_cfg.get("seed") else None)
        )