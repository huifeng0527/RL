from pathlib import Path
from docx import Document

path = Path('manuscripts/methodology_ieee_v2.6_revised_simulation_polished_ieee.docx')
doc = Document(path)
for p in doc.paragraphs:
    if 'formulates it as a reinforcement learning task' in p.text:
        p.text = p.text.replace('formulates it as a reinforcement learning task', 'formulates it as a reinforcement learning (RL) task')
    if 'Reinforcement learning (RL) provides a suitable framework' in p.text:
        p.text = p.text.replace('Reinforcement learning (RL) provides a suitable framework', 'RL provides a suitable framework')
doc.save(path)
print(path.as_posix())
