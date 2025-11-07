import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder_lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        self.decoder_lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config['hidden_size'],
            num_heads=config.get('num_heads', 4),
            dropout=config.get('attention_dropout', 0.1)
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config['output_size'])
        )
    
    def forward(self, encoder_input, decoder_input):
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        attended_output, _ = self.attention(
            decoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1)
        )
        
        combined = torch.cat([decoder_output, attended_output.transpose(0, 1)], dim=-1)
        forecast = self.forecast_head(combined[:, -1, :])
        return forecast