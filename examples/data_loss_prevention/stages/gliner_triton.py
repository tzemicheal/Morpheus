
import torch
import tritonclient.http as httpclient
from gliner import GLiNER
import os
import onnxruntime as ort

class GliNERTritonInference:
    
    def __init__(self, model_source_dir: str,
                 triton_model_name: str = "gliner_bi_encoder",
                 gliner_threshold: float = 0.3,
                 onnx_run: bool = True):
        
        # We load the model locally to use its pre/post-processing functions.
        # The actual heavy inference will be done on Triton.
        print(model_source_dir)
        self.gliner_model = GLiNER.from_pretrained(model_source_dir,
                                                   local_files_only=True,
                                                   onnx_path="model.onnx",
                                                   map_location="cuda",
                                                   load_onnx_model=True
                                                   )
        self.triton_model_name = triton_model_name
        self.gliner_threshold = gliner_threshold
        self.labels_embeddings = torch.tensor([])
        self.onnx_model_path = os.path.join(model_source_dir, "model.onnx")
        
        if onnx_run:
        #    self.ort_session = ort.InferenceSession(self.onnx_model_path)
            self._setup_onnx_runtime(self.onnx_model_path)
        else:
            self.ort_session = None
        
        labels_data = torch.load(os.path.join(model_source_dir,
                                              "labels_embeddings.pt"))
        self.labels_embeddings = labels_data["embeddings"].cpu().numpy()
    
    def _setup_onnx_runtime(self, onnx_model_path):
        
        # Setup ONNX providers for GPU
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 1,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'  # Fallback
        ]
        
        # Load ONNX session with GPU support
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        try:
            self.ort_session = ort.InferenceSession(
                onnx_model_path, 
                session_options,
                providers=providers
            )
            print(f"✓ ONNX session created with providers: "
                  f"{self.ort_session.get_providers()}")
        except Exception as e:
            print(f"⚠️  Failed to create ONNX session with GPU, "
                  f"trying CPU: {e}")
            self.ort_session = ort.InferenceSession(
                onnx_model_path, 
                session_options,
                providers=['CPUExecutionProvider']
            )
    
    def post_process_results(self,  logits_tensor, raw_batch, texts) -> list:
        """
        Post-process the results from the ONNX model.
        """
        onnx_results = self.gliner_model.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            logits_tensor,
            flat_ner=True,
            threshold=self.gliner_threshold,
            multi_label=False,
        )

        # Process results to match the expected format
        all_entities = []
        for i, output in enumerate(onnx_results):
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][i]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append(
                    {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                )
            all_entities.append(entities)
        return all_entities

    def pre_process(self,  texts, labels):
        """
        Pre-process the data for the ONNX model.
        """
        # === 1. PRE-PROCESSING ===
        # if self.labels_embeddings.numel() == 0:
        #     self.labels_embeddings = self.gliner_model.encode_labels(labels)
        
        model_input, raw_batch = self.gliner_model.prepare_model_inputs(
            texts, labels, prepare_entities=False
        )

        # Convert torch tensors to numpy for Triton
        onnx_inputs = {
            "labels_embeddings": self.labels_embeddings,
            "input_ids": model_input["input_ids"].cpu().numpy(),
            "attention_mask": model_input["attention_mask"].cpu().numpy(),
            "words_mask": model_input["words_mask"].cpu().numpy(),
            "text_lengths": model_input["text_lengths"].cpu().numpy(),
            "span_idx": model_input["span_idx"].cpu().numpy(),
            "span_mask": model_input["span_mask"].cpu().numpy(),
        }

        return onnx_inputs, raw_batch

    def process(self, texts: list[str], labels: list[str]):
        """
        Performs full NER pipeline: pre-process, infer, post-process.
        """

        # === 1. PRE-PROCESSING ===
        onnx_inputs, raw_batch = self.pre_process(texts, labels)

        # === 2. TRITON INFERENCE ===
        client = httpclient.InferenceServerClient(url="localhost:8000")

        # Create InferInput objects
        triton_inputs = [
            httpclient.InferInput(
                name, data.shape, httpclient.np_to_triton_dtype(data.dtype)
            )
            for name, data in onnx_inputs.items()
        ]

        # Set data for each input
        for i, name in enumerate(onnx_inputs.keys()):
            triton_inputs[i].set_data_from_numpy(onnx_inputs[name])

        # Request output
        triton_outputs = [httpclient.InferRequestedOutput("output")]

        # Get response
        response = client.infer(self.triton_model_name, inputs=triton_inputs,
                                outputs=triton_outputs)
        logits_np = response.as_numpy("output")
     
        # === 3. POST-PROCESSING ===
       # print("Decoding entities...")
        logits = torch.from_numpy(logits_np)

        return self.post_process_results(logits, raw_batch, texts)
       