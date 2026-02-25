#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stimuli_creation.py
@author: christos

order = (SVO, VSO)
question_for = (V1, V2) #transitive, intransitive 
N1_number = (singular, plural)

* There are no number-incongruent cases 
(e.g: it's always either S-S or P-P never S-P or P-S) because forming 
 a question without providing an answer becomes hard (ποιος/ποιοι).

Prototypes:
    1. Det N1 whom Det N2 V_transitive V_intransitive. 
    1. Det N1 whom V_transitive Det N2  V_intransitive. 
    
Examples:    
    
    1. Ο αθλητής που ο γυμναστής θαυμάζει φεύγει. --> Ποιος θαυμάζει? (V1)
    2. Ο αθλητής που θαυμάζει ο γυμναστής φεύγει. --> Ποιος θαυμάζει? (V1)
    3. Ο αθλητής που ο γυμναστής θαυμάζει φεύγει. --> Ποιος φεύγει? (V2)
    4. Ο αθλητής που θαυμάζει ο γυμναστής φεύγει. --> Ποιος φεύγει? (V2)

Things to counter-balance for: 
    1. Order (as many SVOs as VSOs)
    2. Question_for (as many V1 questions as V2)
    3. N1 number (as many Singular as Plural)
    4. N1,N2 gender (as many masculine N1 as feminine)
    5. N1,N2 congruency (as many M-M, F-F, M-F, F-M)
    6. N1 and N2 must be of a different stem 
    (e.g: forbidden to have: 
       Ο αθλητής που ο αθλητής θαυμάζει φεύγει.
       Ο αθλητής που η αθλήτρια θαυμάζει φεύγει.
    )
"""

# =============================================================================
# MODULES
# =============================================================================
import pandas as pd
import os
import unicodedata

import random
from collections import Counter
from lexicon import words


def norm(s):
    return unicodedata.normalize("NFC", str(s).strip())


def noun_stem(noun_str):
    """
    Return a canonical stem key for a noun so that gendered cognates
    (αθλητής / αθλήτρια) are treated as the same stem.

    Strategy: use the position index in the masculine singular list as the
    stem ID, since all four gender/number lists are aligned by meaning.
    Falls back to the normalised string if the noun isn't found there.

    WARNING: If you add new words to the lexicon, make sure they're added to both M/F lists
    """
    n = norm(noun_str)
    # Try to find the noun's position in any gender/number list; same position
    # across lists = same semantic entry = same stem.
    for gen in ["m", "f"]:
        for num in ["sing", "plur"]:
            lst = [norm(x) for x in words["humans"][gen][num]]
            if n in lst:
                return lst.index(n)   # integer stem ID
    return n   # unknown noun: use string as fallback


# Function to check that N1 and N2 have different stems
def check_n1_n2_stems(N1, N2):
    """
    Return True if N1 and N2 share a stem (cognates or identical).
    Uses position-based stem IDs so cross-gender pairs are caught.
    """
    return noun_stem(N1) == noun_stem(N2)

def pick_n2(N1, N2_gender, N1_number):
    """
    Pick an N2 that does not share a stem with N1.
    Raises RuntimeError if no valid candidate can be found.
    """
    candidates = words["humans"][N2_gender][N1_number]
    valid = [n for n in candidates if not check_n1_n2_stems(N1, n)]
    if not valid:
        raise RuntimeError(
            f"No valid N2 candidate for N1='{N1}' with gender={N2_gender}, number={N1_number}."
        )
    return random.choice(valid)

# Define a class to handle the experiment configuration and calculations
class Experiment:
    def __init__(
        self,
        num_cells,
        presentation_time,
        soa,
        question_display_time,
        response_time,
        seed
    ):
        self.num_cells = num_cells
        self.presentation_time = presentation_time
        self.soa = soa
        self.question_display_time = question_display_time
        self.response_time = response_time
        self.words_per_sentence = 7
        self.seed = seed
        random.seed(seed)

    # Function to generate sentence structures based on given parameters with detailed metadata
    def generate_sentence_structure_with_detailed_metadata(
        self, order, question_for, N1_number, N1_gender, N2_gender
    ):
        """
        Generate sentence structures based on given parameters with updated question format and correct verb placement.

        Args:
        - order: SVO or VSO
        - question_for: V1 (transitive) or V2 (intransitive)
        - N1_number: singular or plural
        - N1_gender: m (masculine) or f (feminine)
        - N2_gender: m (masculine) or f (feminine)

        Returns:
        - A tuple containing the generated sentence, the corresponding question, and the metadata.
        """
        # Get determiners
        det_N1 = words["det"][N1_gender][N1_number][0]
        det_N2 = words["det"][N2_gender][N1_number][0]

        # Get nouns
        N1 = random.choice(words["humans"][N1_gender][N1_number])
        N2 = pick_n2(N1, N2_gender, N1_number)

        # Always use transitive verb for V1 and intransitive verb for V2
        verb_trans = random.choice(words["verbs"]["tran"][N1_number])
        verb_intrans = random.choice(words["verbs"]["intr"][N1_number])

        if order == "SVO":
            sentence = f"{det_N1.capitalize()} {N1} που {det_N2} {N2} {verb_trans} {verb_intrans}."
        else:  # VSO
            sentence = f"{det_N1.capitalize()} {N1} που {verb_trans} {det_N2} {N2} {verb_intrans}."

        # Use 'ποιος' for singular and 'ποιοι' for plural
        question_word = "Ποιος" if N1_number == "sing" else "Ποιοι"

        if question_for == "V1":
            question = f"{question_word} {verb_trans}?"
            correct_response = f"{det_N2.capitalize()} {N2}"
            false_response = f"{det_N1.capitalize()} {N1}"
        else:
            question = f"{question_word} {verb_intrans}?"
            correct_response = f"{det_N1.capitalize()} {N1}"
            false_response = f"{det_N2.capitalize()} {N2}"

        metadata = {
            "order": order,
            "question_for": question_for,
            "N1_number": N1_number,
            "N1_gender": N1_gender,
            "N2_gender": N2_gender,
            "correct_response": correct_response,
            "false_response": false_response,
            "N1_noun":          N1,
            "N2_noun":          N2,
            "V1":               verb_trans,
            "V2":               verb_intrans,
        }

        return sentence, question, metadata

    def generate_cell(self, seen_sentences: set):
        balanced_sentences_detailed_metadata = []
        orders = ["SVO", "VSO"]
        questions_for = ["V1", "V2"]
        numbers = ["sing", "plur"]
        genders = ["m", "f"]
        for order in orders:
            for question_for in questions_for:
                for number in numbers:
                    for N1_gender in genders:
                        for N2_gender in genders:
                            # Make sure sentences are unique
                            for attempt in range(200):
                                sentence, question, metadata = (
                                    self.generate_sentence_structure_with_detailed_metadata(
                                        order, question_for, number, N1_gender, N2_gender
                                    )
                                )
                                if sentence not in seen_sentences:
                                    break
                            seen_sentences.add(sentence)
                            balanced_sentences_detailed_metadata.append((sentence, question, metadata))
                            

        # Shuffle sentences to introduce non-repetition across cells
        random.shuffle(balanced_sentences_detailed_metadata)
        return balanced_sentences_detailed_metadata

    def generate_multiple_cells(self):
        seen_sentences = set()         
        all_cells = []
        for _ in range(self.num_cells):
            cell = self.generate_cell(seen_sentences)
            all_cells.append(cell)

        return all_cells

    def calculate_trial_duration(self):
        # Calculate time components
        time_per_word = self.presentation_time + self.soa  # ms
        time_for_sentence = (
            self.words_per_sentence * time_per_word
        )  # Time for all words in a sentence

        # Calculate total time per trial
        time_per_trial = (
            time_for_sentence
            + self.question_display_time  # Time for the entire sentence
            + self.soa  # Time for question display
            + self.response_time  # SOA after question  # Time for response
        )

        return time_per_trial

    def print_summary(self, all_cells):
        total_sentences = sum(len(cell) for cell in all_cells)
        num_cells = len(all_cells)
        sentences_per_cell = len(all_cells[0]) if all_cells else 0

        # Calculate maximum total duration of the experiment
        time_per_trial = self.calculate_trial_duration()
        max_experiment_duration = (
            total_sentences * time_per_trial
        )  # Total time for all trials

        # Convert duration from ms to seconds and minutes for readability
        max_experiment_duration_seconds = max_experiment_duration / 1000.0
        max_experiment_duration_minutes = (
            max_experiment_duration_seconds / 60.0
        )

        # Count occurrences for each category using separate counters
        order_counter = Counter()
        question_for_counter = Counter()
        n1_number_counter = Counter()
        n1_gender_counter = Counter()
        n2_gender_counter = Counter()

        for cell in all_cells:
            for _, _, metadata in cell:
                order_counter[metadata["order"]] += 1
                question_for_counter[metadata["question_for"]] += 1
                n1_number_counter[metadata["N1_number"]] += 1
                n1_gender_counter[metadata["N1_gender"]] += 1
                n2_gender_counter[metadata["N2_gender"]] += 1

        # Calculate trials per category
        trials_per_category = {
            "Order (SVO/VSO)": {
                k: v // num_cells for k, v in order_counter.items()
            },
            "QuestionFor (V1/V2)": {
                k: v // num_cells for k, v in question_for_counter.items()
            },
            "N1_Number (sing/plur)": {
                k: v // num_cells for k, v in n1_number_counter.items()
            },
            "N1_Gender (m/f)": {
                k: v // num_cells for k, v in n1_gender_counter.items()
            },
            "N2_Gender (m/f)": {
                k: v // num_cells for k, v in n2_gender_counter.items()
            },
        }

        print("Experiment Structure:")
        print(f"  - Presentation time per word: {self.presentation_time} ms")
        print(f"  - Stimulus Onset Asynchrony (SOA): {self.soa} ms")
        print(f"  - Question display time: {self.question_display_time} ms")
        print(f"  - Response time: {self.response_time} ms")
        print(f"  - Words per sentence: {self.words_per_sentence}")
        print("\nSummary of the Experiment:")
        print(f"  - Total number of sentences (trials): {total_sentences}")
        print(f"  - Number of cells: {num_cells}")
        print(f"  - Sentences per cell: {sentences_per_cell}")
        print(
            f"  - Maximum total duration of the experiment: {max_experiment_duration_minutes:.2f} minutes"
        )
        # Print trials per category to ensure everything is counterbalanced
        print("\nTrials per category (counterbalanced):")
        for category, counts in trials_per_category.items():
            print(f"  {category}: {counts}")

    def create_dataframe(self, all_cells):
        # Flatten the list of cells to create a DataFrame
        data = [item for cell in all_cells for item in cell]
        df = pd.DataFrame(data, columns=["Sentence", "Question", "Metadata"])
        # Separate metadata into columns
        df = pd.concat(
            [
                df.drop(["Metadata"], axis=1),
                df["Metadata"].apply(pd.Series),
            ],
            axis=1,
        )
        return df


def validate(df):
    # No duplicate sentences
    n_dupes = df["Sentence"].duplicated().sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate sentences!"

    # No same-stem N1/N2
    bad = df.apply(lambda r: check_n1_n2_stems(r["N1_noun"], r["N2_noun"]), axis=1).sum()
    assert bad == 0, f"Found {bad} rows where N1 and N2 share a stem!"

    # Perfectly balanced 32 cells
    cell_counts = df.groupby(
        ["order", "question_for", "N1_number", "N1_gender", "N2_gender"]
    ).size()
    assert cell_counts.min() == cell_counts.max(), (
        f"Cells not equal-sized: min={cell_counts.min()}, max={cell_counts.max()}"
    )

    print(f"\nValidation passed: {len(df)} sentences, "
          f"{int(cell_counts.min())} per condition cell, no duplicates, no same-stem pairs.")


# %%
# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate counterbalanced Greek sentence stimuli for SVO/VSO word order experiments\n"
            "Total sentences = 32 condition cells × num_cells.\n"
            "  num_cells=16 →  512 sentences  (recommended minimum for 20-fold CV)\n"
            "  num_cells=32 → 1024 sentences\n"
            "  num_cells=4  →  128 sentences  (quick test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num_cells",
        type=int,
        default=32,
        help="Number of experimental cells (default: 32)",
    )
    parser.add_argument(
        "--presentation_time",
        type=int,
        default=200,
        help="Word presentation time in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--soa",
        type=int,
        default=366,
        help="Stimulus Onset Asynchrony in milliseconds (default: 366)",
    )
    parser.add_argument(
        "--question_display_time",
        type=int,
        default=200,
        help="Question display time in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--response_time",
        type=int,
        default=500,
        help="Response time window in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../stimuli",
        help="Output directory for stimuli CSV file (default: ../../stimuli)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="greek_sentences.csv",
        help="Output CSV filename (default: greek_sentences.csv)",
    )
    parser.add_argument("--seed",                  type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    experiment = Experiment(
        num_cells=args.num_cells,
        presentation_time=args.presentation_time,
        soa=args.soa,
        question_display_time=args.question_display_time,
        response_time=args.response_time,
        seed=args.seed
    )

    print("Generating stimuli (seed={args.seed})...")
    all_cells = experiment.generate_multiple_cells()
    experiment.print_summary(all_cells)

    # Create and inspect the DataFrame
    df_all_cells = experiment.create_dataframe(all_cells)
    validate(df_all_cells)

    # Store the dataframe
    path_to_stimuli = args.output_dir
    os.makedirs(path_to_stimuli, exist_ok=True)
    fname = os.path.join(path_to_stimuli, args.output_filename)
    df_all_cells.to_csv(fname)
    print(f"\nStimuli saved to: {fname}")


if __name__ == "__main__":
    main()
