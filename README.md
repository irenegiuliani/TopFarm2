🌬️ TOPFARM-UniFi

TOPFARM-UniFi is an extended version of the TopFarm, the 
plant-level optimization framework developed by DTU Wind Energy.
This version introduces additional modules for offshore applications and advanced layout analysis.


🚀 Overview
TOPFARM-UniFi adds several new capabilities to the original TopFarm framework, including:

🪝 Mooring footprint calculation for taut mooring systems
🌊 Bathymetry data integration for site-specific modeling
⚙️ Workflow customization for research and optimization tasks

These extensions are designed to support floating wind farm design and site layout optimization.


🧩 Installation
Follow the steps below to set up your environment and install TOPFARM UniFi:

	1) Create a conda environment, run in your Linux or Windows terminal: conda create --name topfarm_unifi "python<3.12"

	2) Activate the environment: conda activate topfarm_unifi

	3) Install the customized TopFarm directly from this GitHub repository: pip install git+https://github.com/irenegiuliani/TopFarm2.git

	4) Install dependencies, download the dependencies.yml file from this repository and run: conda env update -f dependencies.yml


🗂️ Repository Structure
TopFarmUniFi/
│
├── topfarm/                     # Core optimization framework
│   └── examples/unifi_examples  # UniFi-specific use cases
│
├── dependencies.yml             # Conda dependencies file
├── README.md                    # This documentation
└── LICENSE                      # License information


🤝 Acknowledgements
This project builds upon the TopFarm framework developed by DTU Wind Energy
Additional contributions and UniFi extensions were developed at Università degli Studi di Firenze.

