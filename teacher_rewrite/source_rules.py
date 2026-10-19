#!/usr/bin/env python3
"""Rules v1.3 (v1.2 from the 378-sentence classification + phrases missed in the 60-turn rewrite review): does a sentence cite content the student cannot see?
Returns (is_source, rule_name). Always-source patterns first, then context-dependent words, then own-file exclusions."""
import re
ALWAYS = [
    ("seed0", r"\bseed[- ]?0\b|\bseed0\b|\bs0\b"),
    ("results_dir", r"results_\w+|_pert\b"),
    ("on_file", r"\bon file\b|\barchived\b"),
    ("memory_file", r"MEMORY\.md|\bmemory (library|file|files|entry|entries|index|dir|directory)\b"),
    ("memory_says", r"\b(the|this|that|task[- ]specific|global|dual[- ]\w+|single[- ]\w+|\w+_\w+) memor(y|ies)\b(?! (usage|footprint|limit|error|budget))|\bmemor(y|ies) (says|said|warns?|notes?|recipe|recommends?|shows?|confirms?|describes?)\b|\baccording to (the )?memor(y|ies)\b|\bper (the )?memor(y|ies)\b|\bin memory\b"),
    ("memories_plural", r"\b(related|relevant|remaining|few|more|other|these|those|two|three|matching|key|task) memories\b|\bmemor(y|ies):\s|\bmemor(y|ies) \(|\(memory"),
    ("recipe_have", r"\b(complete|full|whole|entire|seed[- ]?0|s0) recipe\b|\brecipe (for|of) (this|the) (task|cell)\b"),
    ("memory_capital", r"^\W*memor(y|ies)\b(?! (usage|footprint|limit|error|budget))|\bmemory #\d|\bmemory feedback_\w+|\bfeedback_\w+\.md|\b(crash|moka|basket|drawer|stove|mug|bowl|task|cell) memory\b"),
    ("memo_audit_recipe", r"\bstrategy memo\b|\bthe audit'?s\b|\b(this|that) recipe\b|\bthe recipe (step|line|entry|cmd|command)s?\b"),
    ("memory_named", r"\b\w+-[\w-]+ memory\b|\b(box|can|plate|bottle|cabinet|microwave|wine|cream|ketchup|soup|butter|pudding|drawer|fragile|ordered|dual) memory\b|\bfrom (the )?memor(y|ies)\b|\bfeedback_\w+"),
    ("prior_result", r"\bdocumented\b|\bwinn(ing|er)\b|\bproven\b(?! levers)|\bthe (solution|guidance|baseline|reference corpus)\b|\bfailure mode #?\d|\bthat run\b|\bthe reference run\b"),
    ("they_did", r"\bthey (did|used|released|reached|carried|placed|descended|got|went|grasped|dropped|lifted|opened|closed|pushed|approached)\b"),
    ("recipe_verb", r"\bthe recipe (says|released|placed|picks|uses|used|does|did|goes|moves|has)\b|\b(recipe|audit|memory|reference) explicitly\b"),
    ("author_attempts", r"\b\d+ (prior|archived|previous|failed) attempts\b|\battempts? (#)?\d{2,}\b(?!\s*(step|turn))"),
]
CONTEXT = [
    ("recipe_cited", r"\b(proven|reference|winning|standard|sibling|seed[- ]?0|s0|documented|the solved)\b[^.]{0,25}\brecipes?\b|\brecipes? (says|said|used|uses|shows|places|pre-?positions|did)\b|\bper the recipe\b|\bthe recipe'?s\b"),
    ("audit_cited", r"\b(seed[- ]?0|s0|reference|prior|archived|master|swap s0)\b[^.]{0,20}\baudits?\b|\baudit (says|said|states|shows|confirms|described|notes)\b"),
    ("reference_cited", r"\b(the|this) reference('s)? (says|said|shows|used|uses|recipe|audit|solution|winning|strategy|notes?)\b|\breference (task|solution|strategy|run)\b"),
    ("notes_cited", r"\bper the notes?\b|\bfrom the notes?\b|\bthe notes? (says|said|say|warns?|recommends?)\b|\bstrategy_notes (say|says|of)\b|\bthe \w+ notes\b"),
]
OWN_FILE = r"\b(write|writing|wrote|written|export|exported|save|saving|finaliz\w*|record|recording)\b[^.]{0,30}\b(audit|recipe)s?\b|\b(audit|recipe) (file|json|jsonl)\b.{0,20}\b(write|written|export)"
def classify(sentence):
    s = sentence
    for name, rx in ALWAYS:
        if re.search(rx, s, re.I): return True, name
    for name, rx in CONTEXT:
        if re.search(rx, s, re.I):
            if re.search(OWN_FILE, s, re.I) and name in ("recipe_cited", "audit_cited"): return False, "own_file"
            return True, name
    return False, None
