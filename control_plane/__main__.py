"""
Main entry point for the Agent Control Plane.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from control_plane import ControlPlane, ControlPlaneConfig, DEFAULT_CONFIG


async def main():
    """Run the control plane."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # Load config from file if exists, otherwise use default
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        config = ControlPlaneConfig.from_file(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        config = DEFAULT_CONFIG
        logger.info("Using default configuration")

    # Create and start control plane
    cp = ControlPlane(config)

    try:
        await cp.start()
        logger.info("Control plane started. Press Ctrl+C to stop.")

        # Keep running
        while cp.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await cp.stop()
        logger.info("Control plane stopped")


if __name__ == "__main__":
    asyncio.run(main())
