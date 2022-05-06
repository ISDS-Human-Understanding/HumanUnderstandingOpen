import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(0, 1))

        if mask is not None:
            mask = mask.unsqueeze(1)
            weights = weights.masked_fill(mask == 0, -float('inf'))

        normlized_weights = self.softmax(weights)
        output = torch.matmul(normlized_weights, v)
        output = self.out(output)

        return output, normlized_weights


class MultimodalCrossAttention(nn.Module):
    def __init__(self, audio_config, text_config):
        super().__init__()
        self.audio_args = audio_config
        self.text_args = text_config

        self.conv1d = nn.Conv1d(1, 1, kernel_size=1, stride=self.audio_args.K, padding=0, bias=False)

        self.attention_hidden_size = self.text_args.output_dim
        self.attention = Attention(self.attention_hidden_size)

    def _conv1d(self, output):
        out = self.conv1d(output.unsqueeze(1))
        out = out.squeeze(1)
        return out

    def forward(self, speech_embed, text_embed):
        speech_embed = self._conv1d(speech_embed)
        assert text_embed.shape == speech_embed.shape

        # text to speech cross attention
        text_query = text_embed
        speech_key = speech_embed
        speech_value = speech_embed

        tts_output, tts_normalized_weights = self.attention(text_query, speech_key, speech_value)

        # speech to text cross attention
        speech_query = speech_embed
        text_key = text_embed
        text_value = text_embed

        stt_output, stt_normalized_weights = self.attention(speech_query, text_key, text_value)

        return tts_output, tts_normalized_weights, stt_output, stt_normalized_weights
