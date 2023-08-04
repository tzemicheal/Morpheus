# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from io import StringIO

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage


@pytest.mark.use_python
class TestGraphConstructionStage:

    def test_constructor(self, config: Config, training_file: str):
        from stages.graph_construction_stage import FraudGraphConstructionStage
        stage = FraudGraphConstructionStage(config, training_file)
        assert isinstance(stage._training_data, cudf.DataFrame)

        # The training datafile contains many more columns than this, but these are the four columns
        # that are depended upon in the code
        assert {'client_node', 'index', 'fraud_label', 'merchant_node'}.issubset(stage._column_names)

    def test_process_message(self, dgl: types.ModuleType, config: Config, test_data: dict):
        from stages import graph_construction_stage
        df = test_data['df']

        # The stage wants a csv file from the first 5 rows
        training_data = StringIO(df[0:5].to_csv(index=False))
        stage = graph_construction_stage.FraudGraphConstructionStage(config, training_data)

        # Since we used the first 5 rows as the training data, send the second 5 as inference data
        meta = MessageMeta(cudf.DataFrame(df))
        multi_msg = MultiMessage(meta=meta, mess_offset=5, mess_count=5)
        fgmm = stage._process_message(multi_msg)

        assert isinstance(fgmm, graph_construction_stage.FraudGraphMultiMessage)
        assert fgmm.meta is meta
        assert fgmm.mess_offset == 5
        assert fgmm.mess_count == 5

        assert isinstance(fgmm.graph, dgl.DGLGraph)
        fgmm.graph.check_graph_for_ml(features=True, expensive_check=True)  # this will raise if it doesn't pass
        assert not fgmm.graph.is_directed()

        nodes = fgmm.graph.nodes()
        assert set(nodes) == test_data['expected_nodes']

        edges = fgmm.graph.edges()
        assert set(edges) == test_data['expected_edges']
