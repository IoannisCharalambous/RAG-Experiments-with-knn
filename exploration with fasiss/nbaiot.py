# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""N-BaIoT dataset loader."""


import os
import rarfile
import numpy as np
import pandas as pd
import datasets


_CITATION = """\
@article{DBLP:journals/corr/abs-1805-03409,
  author       = {Yair Meidan and
                  Michael Bohadana and
                  Yael Mathov and
                  Yisroel Mirsky and
                  Dominik Breitenbacher and
                  Asaf Shabtai and
                  Yuval Elovici},
  title        = {N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep
                  Autoencoders},
  journal      = {CoRR},
  volume       = {abs/1805.03409},
  year         = {2018},
  url          = {http://arxiv.org/abs/1805.03409},
  eprinttype    = {arXiv},
  eprint       = {1805.03409},
  timestamp    = {Mon, 13 Aug 2018 16:49:04 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1805-03409.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""
_DESCRIPTION = """\
An intrusion detection dataset that focuses on IoT botnet attacks.
"""
_HOMEPAGE = "https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot"
_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0) license"
_URL = 'https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip'
_ATTACK_NAMES = ['benign_traffic', 'combo', 'junk', 'mirai-ack', 'mirai-scan', 'mirai-syn', 'mirai-udp', 'mirai-udpplain', 'scan', 'tcp', 'udp']
_DEVICE_NAMES = [
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam", "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera"
]


class NBAIOTDataset(datasets.GeneratorBasedBuilder):
    """N-BaIoT intrusion detection."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features({
                "features": datasets.Sequence(feature=datasets.Value("float32"), length=115),
                "attack": datasets.ClassLabel(len(_ATTACK_NAMES), names=_ATTACK_NAMES),
                "device": datasets.ClassLabel(len(_DEVICE_NAMES), names=_DEVICE_NAMES),
            }),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        for device in _DEVICE_NAMES:
            # First load in the benign traffic
            all_data = pd.read_csv(f"{filepath}/{device}/benign_traffic.csv").values
            attacks = np.repeat("benign_traffic", len(all_data))
            # Then the standard attacks
            attacks_rar = rarfile.RarFile(f"{filepath}/{device}/gafgyt_attacks.rar")
            for fileinfo in attacks_rar.infolist():
                with attacks_rar.open(fileinfo.filename) as f:
                    data = pd.read_csv(f).values
                attacks = np.concatenate((attacks, np.repeat(f.name.replace(".csv", ""), len(data))))
                all_data = np.concatenate((all_data, data))
            # And, if present, the Mirai attacks
            if device not in ["Ennio_Doorbell", "Samsung_SNH_1011_N_Webcam"]:
                mirai_rar = rarfile.RarFile(f"{filepath}/{device}/mirai_attacks.rar")
                for fileinfo in mirai_rar.infolist():
                    with mirai_rar.open(fileinfo.filename) as f:
                        data = pd.read_csv(f).values
                    attacks = np.concatenate((attacks, np.repeat("mirai-" + f.name.replace(".csv", ""), len(data))))
                    all_data = np.concatenate((all_data, data))
            # Create the train-test split
            rng = np.random.default_rng(round(np.pi**(np.pi * 100)))
            train = rng.uniform(size=len(all_data)) < 0.85
            all_data = all_data[train if split == "train" else ~train]
            # Finally yield the data
            for key, (data, attack) in enumerate(zip(all_data, attacks)):
                yield key, {
                    "features": data,
                    "attack": attack,
                    "device": device,
                }