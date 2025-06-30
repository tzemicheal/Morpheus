# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import time

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg
import pandas as pd
import numpy as np

@register_stage("risk-scorer")
class RiskScorer(GpuAndCpuMixin, ControlMessageStage):
    """Analyzes findings to calculate risk scores and metrics"""

    DEFAULT_TYPE_WEIGHTS = {
        "password": 85,
        "credit_card_number": 90,
        "ssn": 95,
        "street_address": 60,
        "email": 40,
        "phone_number": 45,
        "ipv4": 30,
        "ipv6": 30,
        "date": 20,
        "date_time": 20,
        "time": 20,
        "api_key": 80,
        "customer_id": 65,
        "health_plan_beneficiary_number": 75,
        "medical_record_number": 75
    }

    _NEW_COLUMNS = {
        "risk_score": 0,
        "risk_level": '',
        "highest_confidence": 0.0,
        "num_minimal": 0,
        "num_low": 0,
        "num_medium": 0,
        "num_high": 0,
        "num_critical": 0
    }

    def __init__(self,
                 config: Config,
                 *,
                 findings_column: str,
                 type_weights: dict[str, int] | None = None,
                 default_weight: int = 50):
        """Initialize with configuration for risk scoring"""
        super().__init__(config)

        if type_weights is not None:
            self.type_weights = type_weights
        else:
            self.type_weights = self.DEFAULT_TYPE_WEIGHTS.copy()

        # Default weight if type not in dictionary
        self.default_weight = default_weight

        self._findings_column = findings_column
        self._df_class = get_df_class(config.execution_mode)
        self._df_pkg = get_df_pkg(config.execution_mode)
        self._elapsed_time_secs = 0.0
        self._group_cols = [self._findings_column, "data_types_found"] + list(self._NEW_COLUMNS.keys())

    @property
    def name(self) -> str:
        return "risk-scorer"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @staticmethod
    def _risk_score_to_level(risk_score: int) -> str:
        """Convert risk score to risk level string"""
        if risk_score >= 80:
            return "critical"

        if risk_score >= 60:
            return "high"

        if risk_score >= 40:
            return "medium"

        if risk_score >= 20:
            return "low"

        return "minimal"
    
    @staticmethod
    def _risk_score_to_level_vectorized(scores: pd.Series) -> pd.Series:
        """Vectorized risk level calculation"""
        return pd.cut(scores, 
                     bins=[0, 30, 50, 70, 90, 101], 
                     labels=["minimal", "low", "medium", "high", "critical"])
    
    def _score_fn(self,
                  group_df: DataFrameType,
                  *,
                  findings_column: str,
                  type_weights: dict[str, int],
                  default_weight: int,
                  df_class: type) -> DataFrameType | None:

        findings = group_df[findings_column].to_pandas()
        
        if findings is None or findings.empty:
            return None
        
        findings_df = pd.DataFrame({
            'original_index': findings.index,
            'findings': findings
        })
        
        exploded_df = findings_df.explode('findings').dropna(subset=['findings'])
        if exploded_df.empty:
            return None
        
        # Step 2: Handle string splitting and flattening in vectorized way
        def flatten_finding(finding):
            """Flatten a single finding (string or dict)"""
            if isinstance(finding, str):
                return [s.strip() for s in finding.split(',')]
            else:
                return [finding] if not isinstance(finding, list) else finding
        
        # Apply flattening and explode again
        exploded_df['flattened_findings'] = exploded_df['findings'].apply(flatten_finding)
        final_df = exploded_df.explode('flattened_findings').dropna(subset=['flattened_findings'])
        
        if final_df.empty:
            return None
        
        # Step 3: Separate dict and string findings using vectorized operations
        is_dict_mask = final_df['flattened_findings'].apply(lambda x: isinstance(x, dict))
        
        # Process dict findings (from GliNER processor)
        dict_findings = final_df[is_dict_mask].copy()
        if not dict_findings.empty:
            dict_findings['data_type'] = dict_findings['flattened_findings'].apply(lambda x: x['label'])
            dict_findings['confidence'] = dict_findings['flattened_findings'].apply(lambda x: x['score'])
        
        # Process string findings (bypassed)
        str_findings = final_df[~is_dict_mask].copy()
        if not str_findings.empty:
            str_findings['data_type'] = str_findings['flattened_findings']
            str_findings['confidence'] = 1.0
        
        # Combine both types
        if not dict_findings.empty and not str_findings.empty:
            processed_df = pd.concat([dict_findings, str_findings], ignore_index=True)
        elif not dict_findings.empty:
            processed_df = dict_findings
        else:
            processed_df = str_findings
        
        # Step 4: Vectorized weight mapping and score calculation
        processed_df['weight'] = processed_df['data_type'].map(type_weights).fillna(default_weight)
        processed_df['weighted_score'] = processed_df['weight'] * processed_df['confidence']
        
        # Step 5: Vectorized risk level calculation
        processed_df['risk_level'] = processed_df['weight'].apply(self._risk_score_to_level)
        
        
        # Step 6: Aggregate results using pandas groupby operations
        agg_results = processed_df.groupby('original_index').agg({
            'weighted_score': 'sum',
            'confidence': 'max',
            'data_type': lambda x: sorted(set(x)),
            'risk_level': lambda x: pd.Series(x).value_counts().to_dict()
        }).reset_index()
        
        # Step 7: Calculate final metrics
        findings_count = processed_df.groupby('original_index').size()
        agg_results = agg_results.merge(findings_count.rename('findings_count'), left_on='original_index', right_index=True)
        
        # Calculate normalized risk score
        agg_results['risk_score'] = (agg_results['weighted_score'] / agg_results['findings_count']).round().clip(0, 100).astype(int)
        agg_results['risk_level_final'] = agg_results['risk_score'].apply(lambda x: self._risk_score_to_level(x).title())
        
        # Step 8: Extract score counts in vectorized way
        def extract_score_counts(risk_level_dict):
            return {
                "low": risk_level_dict.get("low", 0),
                "medium": risk_level_dict.get("medium", 0), 
                "high": risk_level_dict.get("high", 0),
                "critical": risk_level_dict.get("critical", 0),
                "minimal": risk_level_dict.get("minimal", 0)
            }
        
        score_counts = agg_results['risk_level'].apply(extract_score_counts)
        
        # Step 9: Prepare final result DataFrame
        result_data = {
            "risk_score": agg_results['risk_score'].iloc[0],
            "risk_level": agg_results['risk_level_final'].iloc[0],
            "data_types_found": [agg_results['data_type'].iloc[0]],
            "highest_confidence": agg_results['confidence'].iloc[0],
            findings_column: [processed_df['flattened_findings'].tolist()]
        }
        
        
        # Add score counts to result
        score_counts_dict = score_counts.iloc[0]
        for level, count in score_counts_dict.items():
            result_data[f"num_{level}"] = count
        
        return df_class(result_data)
        
        
    
    def _score_fn_vc(self,
                  group_df: DataFrameType,
                  *,
                  findings_column: str,
                  type_weights: dict[str, int],
                  default_weight: int,
                  df_class: type) -> DataFrameType | None:

        findings = group_df[findings_column].to_pandas()

        if findings is None or findings.empty:
            return None

        flat_findings = []
        original_indices = []
        
        # for finding in findings:
        #     if isinstance(finding, str):
        #         flat_findings.extend(s.strip() for s in finding.split(','))
        #     else:
        #         flat_findings.extend(finding)
        
        for idx, finding_list in findings_series.items():
            if not finding_list:
                continue
                
            # Flatten findings
            flattened = []
            for finding in finding_list:
                if isinstance(finding, str):
                    flattened.extend(s.strip() for s in finding.split(','))
                else:
                    flattened.append(finding)
            
            all_findings.extend(flattened)
            original_indices.extend([idx] * len(flattened))
                
            original_indices.extend([idx] * len(flattened))

        findings = flat_findings

        if len(findings) == 0:
            return None
        # Process all findings at once
       
        # Create weight mapping series for fast lookup
        weights_series = pd.Series(type_weights)
        
        # Create processing DataFrame
        process_df = pd.DataFrame({
            'original_index': original_indices,
            'finding': findings
        })
        
        # Vectorized processing of findings
        is_dict = process_df['finding'].apply(lambda x: isinstance(x, dict))
        
         # Extract data types and confidences vectorized
        process_df['data_type'] = np.where(
            is_dict,
            process_df['finding'].apply(lambda x: x.get('label', '') if isinstance(x, dict) else ''),
            process_df['finding'].astype(str)
        )
        
        process_df['confidence'] = np.where(
            is_dict,
            process_df['finding'].apply(lambda x: x.get('score', 1.0) if isinstance(x, dict) else 1.0),
            1.0
        )
        # Vectorized weight mapping
        process_df['weight'] = process_df['data_type'].map(weights_series).fillna(default_weight)
        
        # Vectorized score calculation
        process_df['weighted_score'] = process_df['weight'] * process_df['confidence']
        process_df['risk_level'] = self._risk_score_to_level_vectorized(process_df['weight'])
        
        # Aggregation using optimized groupby
        agg_data = process_df.groupby('original_index').agg({
            'weighted_score': 'sum',
            'confidence': 'max',
            'data_type': lambda x: sorted(set(x)),
            'finding': 'count',  # Count for normalization
            'risk_level': lambda x: x.value_counts().to_dict()
        })
        
        # Final calculations
        agg_data['risk_score'] = (agg_data['weighted_score'] / agg_data['finding']).round().clip(0, 100).astype(int)
        agg_data['final_risk_level'] = self._risk_score_to_level_vectorized(agg_data['risk_score'])
        
        
        
        # Extract the first (and likely only) result
        #if len(agg_data) > 0:
        from IPython import embed; embed()
        first_result = agg_data.iloc[0]
        
        # Extract score counts
        score_counts = first_result['risk_level'] if isinstance(first_result['risk_level'], dict) else {}
        
        result_data = {
            "risk_score": first_result['risk_score'],
            "risk_level": first_result['final_risk_level'].title() if hasattr(first_result['final_risk_level'], 'title') else str(first_result['final_risk_level']).title(),
            "data_types_found": [first_result['data_type']],
            "highest_confidence": first_result['confidence'],
            findings_column: [findings]
        }
        
        # Add individual score counts
        for level in ["low", "medium", "high", "critical", "minimal"]:
            result_data[f"num_{level}"] = score_counts.get(level, 0)
        
        # df_data.update({f"num_{level}": count for (level, count) in score_counts.items()})
        return df_class(result_data)
            
            #return df_class(result_data)
        
        # # Calculate total weighted score
        # total_score = 0
        # score_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0, "minimal": 0}

        # data_types_found = set()
        # highest_confidence = 0

        # for finding in findings:
        #     # When `finding` is a dict it came from the GliNER processor, if not then it was bypassed
        #     if isinstance(finding, dict):
        #         data_type: str = finding["label"]

        #         # Adjust by confidence
        #         confidence = finding["score"]
        #     else:
        #         data_type = finding
        #         confidence = 1.0

        #     data_types_found.add(data_type)

        #     # Get weight for this data type
        #     weight = type_weights.get(data_type, default_weight)

        #     if confidence > highest_confidence:
        #         highest_confidence = confidence

        #     weighted_score = weight * confidence
        #     total_score += weighted_score

        #     # Count by severity
        #     score_counts[RiskScorer._risk_score_to_level(weight)] += 1

        # # Normalize to 0-100 scale with diminishing returns for many findings
        # max_score = 100

        # # Calculate normalized risk score
        # risk_score = round(min(max_score, total_score / len(findings)))

        # # Determine risk level from score
        # risk_level = RiskScorer._risk_score_to_level(risk_score).title()

        # df_data = {
        #     "risk_score": risk_score,
        #     "risk_level": risk_level,
        #     "data_types_found": [sorted(data_types_found)],
        #     "highest_confidence": highest_confidence,
        #     findings_column: [findings]
        # }

        # df_data.update({f"num_{level}": count for (level, count) in score_counts.items()})

        # return df_class(df_data)

    def score(self, msg: ControlMessage) -> ControlMessage:
        """
        Calculate risk scores based on findings
        """

        t1 = time.time()
        with msg.payload().mutable_dataframe() as df:
            df = df.assign(**self._NEW_COLUMNS)
            df["data_types_found"] = self._df_pkg.Series(index=df.index, dtype=self._df_pkg.core.dtypes.ListDtype)
            groups = df.groupby(["original_source_index"], as_index=False)
            score_fn = functools.partial(self._score_fn,
                                         findings_column=self._findings_column,
                                         type_weights=self.type_weights,
                                         default_weight=self.default_weight,
                                         df_class=self._df_class)
            result_df = groups[self._group_cols].apply(score_fn)
            #from IPython import embed; embed()
            result_df = result_df.rename(columns={'index': 'original_source_index'})

        msg.payload(MessageMeta(result_df))

        t2 = time.time()
        self._elapsed_time_secs += t2 - t1
       # self._on_completed()
        return msg

    def _on_completed(self) -> None:
        print(f"RiskScorer completed in {self._elapsed_time_secs:.2f} seconds")

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.score), ops.on_completed(self._on_completed))
        builder.make_edge(input_node, node)
        return node