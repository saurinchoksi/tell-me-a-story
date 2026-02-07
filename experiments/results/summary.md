# Initial Prompt Experiment Results

Generated: 2026-02-05T16:08:24.351956

Audio: `sessions/00000000-000000/audio.m4a`
Model: `mlx-community/whisper-large-v3-mlx`

## Run Configuration

| Run | Label | Initial Prompt | Time (s) |
|-----|-------|----------------|----------|
| 0 | baseline | None | 53.0 |
| 1 | vocab_list | "Pandavas, Kauravas, Yudhishthi..." | 62.0 |
| 2 | natural_sentence | "This is a story about the Maha..." | 87.6 |

## Name Detection Results

| Name | baseline | vocab_list | natural_sentence |
|------|------|------|------|
| Pandavas | 10 (fondos) | - | 9 (Pandava) |
| Kauravas | 4 (goros) | 4 (Kaurava) | 3 (Kaurava) |
| Yudhishthira | 10 (Yudister.) | 3 (Yudhishthira) | 4 (Yudhishthira) |
| Duryodhana | 8 (Duryodhan,) | 8 (Duryodhana,) | 7 (Duryodhana,) |
| Dhritarashtra | 1 (Dhrashtra) | 1 (Dhrashtra) | - |
| Pandu | - | - | - |
| Bhima | - | - | - |
| Arjuna | - | - | - |
| Draupadi | - | - | - |
| Karna | - | - | - |
| Krishna | - | - | - |
| Mahabharata | - | - | - |

## Observations

_To be filled in after reviewing results._

## Full Variant Details

### Run 0: baseline

- **Pandavas**: ['fondos', 'Fondo,', 'Fondos.', 'fondos,', 'Fondo.', 'Fondo', 'Fondos']
- **Kauravas**: ['goros', 'goros,']
- **Yudhishthira**: ['Yudister.', 'Yudister', 'Yudhisthir', "Yudhisthir's", 'Yudhisthir,']
- **Duryodhana**: ['Duryodhan,', 'Duryodhan', 'Duryodhan.']
- **Dhritarashtra**: ['Dhrashtra', 'Dhrashtra,']

### Run 1: vocab_list

- **Kauravas**: ['Kaurava', 'Kauravas,', 'Kauravas']
- **Yudhishthira**: ['Yudhishthira', 'Yudhishthira.']
- **Duryodhana**: ['Duryodhana,', 'Duryodhana', 'Duryodhana.']
- **Dhritarashtra**: ['Dhrashtra', 'Dhrashtra,']

### Run 2: natural_sentence

- **Pandavas**: ['Pandava', 'Pando.', 'Pandavas', 'Pando', 'Pandavas,', 'Pando,', "Pando's."]
- **Kauravas**: ['Kaurava', 'Kauravas,', 'Kauravas']
- **Yudhishthira**: ['Yudhishthira', 'Yudhishthira...', 'Yudhishthira.']
- **Duryodhana**: ['Duryodhana,', 'Duryodhana', 'Duryodhana.']
