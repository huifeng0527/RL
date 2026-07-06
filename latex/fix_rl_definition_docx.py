from pathlib import Path
from docx import Document

path = Path('manuscripts/methodology_ieee_v2.6_revised_simulation_polished_ieee.docx')
doc = Document(path)
for p in doc.paragraphs:
    if 'Reinforcement learning provides a suitable framework' in p.text:
        p.text = p.text.replace('Reinforcement learning provides a suitable framework', 'Reinforcement learning (RL) provides a suitable framework')
    if 'reinforcement learning provides a suitable framework' in p.text:
        p.text = p.text.replace('reinforcement learning provides a suitable framework', 'reinforcement learning (RL) provides a suitable framework')
doc.save(path)
print(path.as_posix())
