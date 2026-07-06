from pathlib import Path
from docx import Document

path = Path('manuscripts/methodology_ieee_v2.6_revised_simulation_polished_ieee.docx')
doc = Document(path)
replacements = {
    'The league result therefore suggests that opponent accumulation improves the lower tail of performance, rather than merely optimizing for an easy subset of virtual patients.':
    'The league result therefore suggests that opponent accumulation improves lower-tail performance, rather than merely optimizing for an easy subset of virtual patients.',
    'The table reports TIS / ZPD coverage / episode length / catch rate. The scripted-only baseline behaves as expected: it remains relatively effective on the scripted hand but is frequently caught by the learned agent. The single-agent baseline shows the opposite pattern, performing well against the learned hand while degrading on the scripted controller. These complementary failures indicate that training against a single opponent type can produce brittle specialization. The league-trained robot avoids the strongest form of either failure mode and provides the most balanced behavior across the two automated test conditions. Mouse-controlled human-in-the-loop evaluation will be reported after collection.':
    'Table II uses the format TIS / ZPD coverage / episode length / catch rate. The scripted-only baseline behaves as expected: it remains relatively effective on the scripted hand but is frequently caught by the learned agent. The single-agent baseline shows the opposite pattern, performing well against the learned hand while degrading on the scripted controller. These complementary failures indicate that training against a single opponent type can produce brittle specialization. The league-trained robot avoids the strongest form of either failure mode and provides the most balanced behavior across the two automated test conditions. Mouse-controlled human-in-the-loop evaluation is treated as a separate validation stage.',
    'Future work should therefore study loss weighting, prediction horizon, and curriculum timing to better translate representation quality into policy improvement.The study also has limitations.':
    'Future work should therefore study loss weighting, prediction horizon, and curriculum timing to better translate representation quality into policy improvement. The study also has limitations.',
}
for p in doc.paragraphs:
    text = p.text.strip()
    if text in replacements:
        p.text = replacements[text]
    else:
        for old, new in replacements.items():
            if old in p.text:
                p.text = p.text.replace(old, new)
doc.save(path)
print(path.as_posix())
