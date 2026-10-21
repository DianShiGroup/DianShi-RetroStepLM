# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['nvidia_resiliency_ext',
 'nvidia_resiliency_ext.checkpointing',
 'nvidia_resiliency_ext.checkpointing.async_ckpt',
 'nvidia_resiliency_ext.checkpointing.local',
 'nvidia_resiliency_ext.checkpointing.local.ckpt_managers',
 'nvidia_resiliency_ext.checkpointing.local.replication',
 'nvidia_resiliency_ext.fault_tolerance',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.agent',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.agent.server',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.events',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.metrics',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing.errors',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing.subprocess_handler',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.rendezvous',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.timer',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.utils',
 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.utils.data',
 'nvidia_resiliency_ext.inprocess',
 'nvidia_resiliency_ext.inprocess.tools',
 'nvidia_resiliency_ext.ptl_resiliency',
 'nvidia_resiliency_ext.shared_utils',
 'nvidia_resiliency_ext.straggler']

package_data = \
{'': ['*']}

install_requires = \
['defusedxml',
 'nvidia-ml-py>=12.570.86',
 'packaging',
 'psutil>=6.0.0',
 'pynvml>=12.0.0',
 'pyyaml',
 'torch>=2.3.0']

entry_points = \
{'console_scripts': ['ft_launcher = '
                     'nvidia_resiliency_ext.fault_tolerance.launcher:main']}

setup_kwargs = {
    'name': 'nvidia-resiliency-ext',
    'version': '0.3.0',
    'description': 'NVIDIA Resiliency Package',
    'long_description': '# NVIDIA Resiliency Extension\n\nThe NVIDIA Resiliency Extension (NVRx) integrates multiple resiliency-focused solutions for PyTorch-based workloads.\n\n## Core Components and Capabilities\n\n- **Fault Tolerance**\n  - Detection of hung ranks.  \n  - Restarting training in-job, without the need to reallocate SLURM nodes.\n\n- **In-Process Restarting**\n  - Detecting failures and enabling quick recovery.\n\n- **Async Checkpointing**\n  - Providing an efficient framework for asynchronous checkpointing.\n\n- **Local Checkpointing**\n  - Providing an efficient framework for local checkpointing.\n\n- **Straggler Detection**\n  - Monitoring GPU and CPU performance of ranks.  \n  - Identifying slower ranks that may impede overall training efficiency.\n\n- **PyTorch Lightning Callbacks**\n  - Facilitating seamless NVRx integration with PyTorch Lightning.\n\n## Installation\n\n### From sources\n- `git clone https://github.com/NVIDIA/nvidia-resiliency-ext`\n- `cd nvidia-resiliency-ext`\n- `pip install .`\n\n\n### From PyPI wheel\n- `pip install nvidia-resiliency-ext`\n\n### Platform Support\n\n| Category             | Supported Versions / Requirements                                          |\n|----------------------|----------------------------------------------------------------------------|\n| Architecture         | x86_64, arm64                                                              |\n| Operating System     | Ubuntu 22.04, 24.04                                                        |\n| Python Version       | >= 3.10, < 3.13                                                            |\n| PyTorch Version      | >= 2.3.1 (injob & chkpt), 2.5.1 & 2.6.0 (inprocess)                        |\n| CUDA & CUDA Toolkit  | >= 12.5 (12.8 required for GPU health check)                               |\n| NVML Driver          | >= 535 (570 required for GPU health check)                                 |\n| NCCL Version         | >= 2.21.5 (injob & chkpt), >= 2.21.5 and <= 2.22.3 or 2.26.2 (inprocess)   |\n\n## Usage\n\nFor detailed documentation and usage information about each component, please refer to the https://nvidia.github.io/nvidia-resiliency-ext/.\n',
    'author': 'NVIDIA Corporation',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/NVIDIA/nvidia-resiliency-ext',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.10',
}
from cupti_build import *
build(setup_kwargs)

setup(**setup_kwargs)
