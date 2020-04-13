// Configuraiton for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).
{
  "dataset_reader": {
    "type": "pwc_linker",
    "token_indexers": {
      "token_characters": {
        type: "characters",
        min_padding_length: 5
      },
      "bert": {
        type: "bert-pretrained",
        pretrained_model: std.extVar("BERT_VOCAB"),
        do_lowercase: std.extVar("IS_LOWERCASE"),
        use_starting_offsets: true,
        truncate_long_sequences : false
      }
    },
    "tokenizer": {
      "type" : "word",
      "word_splitter" : {
        "type" : "just_spaces"
      },
    }
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),
  "model": {
    "type": "entity_linker",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
          "bert": ["bert", "bert-offsets"],
          "token_characters": ["token_characters"],
      },
      "token_embedders": {
          "bert": {
              "type": "bert-pretrained",
              "pretrained_model": std.extVar("BERT_WEIGHTS"),
              "requires_grad" : "11,10"
          },
          "token_characters": {
              "type": "character_encoding",
              "embedding": {
                "embedding_dim": 16
              },
              "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 128,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu"
              }
          }
      } 
    },
    "premise_encoder" : {
      "type": "lstm",
        "input_size": 768 + 128,
        "hidden_size": 100,
        "bidirectional": true
    },
    "attend_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "compare_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  "iterator": {
    "type": "bucket_sample",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": 50
  },

  "trainer": {
    "num_epochs": 140,
    "patience": 10,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
    "grad_clipping": 5.0,
    "validation_metric": "+f1",
    "optimizer": {
      "type": "adagrad"
    }
  }
}