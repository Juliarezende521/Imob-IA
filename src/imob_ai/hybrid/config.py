"""Configuração congelada da busca híbrida."""

RRF_K = 60
WEIGHT_CANDIDATES = tuple(value / 10 for value in range(0, 11))

# Selecionado exclusivamente nas 20 consultas de development.
HYBRID_LEXICAL_WEIGHT = 0.9

# peso lexical, MRR, Recall@5, Precision@5, Hit@5, acerto de ausência
CALIBRATION_RESULTS = (
    (0.0, 0.850, 0.705, 0.635, 0.882, 0.667),
    (0.1, 0.851, 0.705, 0.635, 0.882, 0.667),
    (0.2, 0.856, 0.705, 0.635, 0.882, 0.667),
    (0.3, 0.871, 0.764, 0.647, 0.941, 0.667),
    (0.4, 0.897, 0.793, 0.659, 1.000, 0.667),
    (0.5, 0.897, 0.793, 0.659, 1.000, 0.667),
    (0.6, 0.902, 0.793, 0.659, 1.000, 0.667),
    (0.7, 0.902, 0.764, 0.647, 1.000, 0.667),
    (0.8, 0.902, 0.764, 0.647, 1.000, 0.667),
    (0.9, 0.922, 0.764, 0.647, 1.000, 0.667),
    (1.0, 0.917, 0.764, 0.647, 1.000, 0.667),
)
