#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# Greek lexicon (within-clause verb movement) 
# =============================================================================
"""

genders = ["m", "f"]
numbers = ["sing", "plur"]

# =============================================================================
# INITIALIZE CONTAINERS
# =============================================================================
# ~~~~~~
# DETERMINERS, HUMANS AND ADJECTIVES
# ~~~~~~
humans, adj, det = [{} for i in range(0, 3)]
for gender in genders:
    humans[gender] = {}
    adj[gender] = {}
    det[gender] = {}
    for number in numbers:
        humans[gender][number] = {}
        adj[gender][number] = {}
        det[gender][number] = {}

# ~~~~~~
# INANIMATE OBJECTS
# ~~~~~~
inanimate = [
    "τραπέζι",
    "καρέκλα",
    "πολυθρόνα",
    "καναπές",
    "καλοριφέρ",
    "τασάκι",
]
# ~~~~~~
# ADVERB
# ~~~~~~
adv = ["που"]

# =============================================================================
# POPULATE CONTAINERS
# =============================================================================

# ~~~~~~
# HUMANS
# ~~~~~~

human_sing_masculine = [
    "δάσκαλος",
    "μαθητής",
    "φοιτητής",
    "γυμναστής",
    "καθηγητής",
    "λογιστής",
    "μεταφραστής",
    "σχεδιαστής",
    "πωλητής",
    "κομμωτής",
    "νοσοκόμος",
    "χορευτής",
    "αθλητής",
    "μάγειρας",
    "εργάτης",
    "υποψήφιος",
]
human_plur_masculine = [
    "δάσκαλοι",
    "μαθητές",
    "φοιτητές",
    "γυμναστές",
    "καθηγητές",
    "λογιστές",
    "μεταφραστές",
    "σχεδιαστές",
    "πωλητές",
    "κομμωτές",
    "νοσοκόμοι",
    "χορευτές",
    "αθλητες",
    "μάγειρες",
    "εργάτες",
    "υποψήφιοι",
]


human_sing_feminine = [
    "δασκάλα",
    "μαθήτρια",
    "φοιτήτρια",
    "γυμνάστρια",
    "καθηγήτρια",
    "λογίστρια",
    "μεταφράστρια",
    "σχεδιάστρια",
    "πωλήτρια",
    "κομμώτρια",
    "νοσοκόμα",
    "χορεύτρια",
    "αθλήτρια",  # ?
    "μαγείρισσα",
    "εργάτρια",
    "υποψήφια",
]


human_plur_feminine = [
    "δασκάλες",
    "μαθήτριες",
    "φοιτήτριες",
    "γυμνάστριες",
    "καθηγήτριες",
    "λογίστριες",
    "μεταφράστριες",
    "σχεδιάστριες",
    "πωλήτριες",
    "κομμώτριες",
    "νοσοκόμες",
    "χορεύτριες",
    "αθλήτριες",  # ?
    "μαγείρισσες",
    "εργάτριες",
    "υποψήφιες",
]


humans["m"]["sing"] = human_sing_masculine
humans["m"]["plur"] = human_plur_masculine


humans["f"]["sing"] = human_sing_feminine
humans["f"]["plur"] = human_plur_feminine


# ~~~~~~
# DET
# ~~~~~~
det_sing_masculine = ["ο"]
det_sing_feminime = ["η"]
det_plur_masculine = ["οι"]
det_plur_feminime = ["οι"]


det["m"]["sing"] = det_sing_masculine
det["m"]["plur"] = det_plur_masculine


det["f"]["sing"] = det_sing_feminime
det["f"]["plur"] = det_plur_feminime


# ~~~~~~
# VERBS
# ~~~~~~
verb_intr_sing = [
    "φεύγει",
    "κλαίει",
    "σκέφτεται",  # ?
    "γελάει",
    "βήχει",
    "πονάει",
]

verb_intr_plur = [
    "φεύγουν",
    "κλαίνε",
    "σκέφτονται",  # ?
    "γελάνε",
    "βήχουν",
    "πονάνε",
]


verb_tran_sing = [
    "συμπαθεί",
    "αντιπαθεί",
    "αγαπάει",  # ?
    "μισεί",
    "γνωρίζει",
    "θαυμάζει",
    # "προτιμά",
    "εκτιμά",
]

verb_tran_plur = [
    "συμπαθούν",
    "αντιπαθούν",
    "αγαπούν",  # ?
    "μισούν",
    "γνωρίζουν",
    "θαυμάζουν",
    # "προτιμούν",
    "εκτιμούν",
]

verbs = {}
for tr in ["intr", "tran"]:
    verbs[tr] = {}

verbs["intr"]["sing"] = verb_intr_sing
verbs["intr"]["plur"] = verb_intr_plur

verbs["tran"]["sing"] = verb_tran_sing
verbs["tran"]["plur"] = verb_tran_plur


words = {}
words["verbs"] = verbs
words["humans"] = humans
words["det"] = det
words["adj"] = adj
words["adv"] = adv
words["inanimate"] = inanimate
