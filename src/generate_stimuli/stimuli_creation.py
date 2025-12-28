#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import random
from collections import Counter
from lexicon import words


# Function to check that N1 and N2 have different stems
def check_n1_n2_stems(n1_index, n2_index, n_entries):
    """
    Ensure that N1 and N2 are different by generating a new index for N2 if necessary.
    """
    if n1_index == n2_index:
        n2_index = random.choice(
            [i for i in range(n_entries) if i != n1_index]
        )
    return n2_index


# Define a class to handle the experiment configuration and calculations
class Experiment:
    def __init__(
        self,
        num_cells,
        presentation_time,
        soa,
        question_display_time,
        response_time,
    ):
        self.num_cells = num_cells
        self.presentation_time = presentation_time
        self.soa = soa
        self.question_display_time = question_display_time
        self.response_time = response_time
        self.words_per_sentence = 7

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
        N2_index = check_n1_n2_stems(
            words["humans"][N1_gender][N1_number].index(N1),
            random.randint(0, len(words["humans"][N2_gender][N1_number]) - 1),
            len(words["humans"][N2_gender][N1_number]),
        )
        N2 = words["humans"][N2_gender][N1_number][N2_index]

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
        }

        return sentence, question, metadata

    def generate_cell(self):
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
                            sentence, question, metadata = (
                                self.generate_sentence_structure_with_detailed_metadata(
                                    order,
                                    question_for,
                                    number,
                                    N1_gender,
                                    N2_gender,
                                )
                            )
                            balanced_sentences_detailed_metadata.append(
                                (sentence, question, metadata)
                            )

        # Shuffle sentences to introduce non-repetition across cells
        random.shuffle(balanced_sentences_detailed_metadata)
        return balanced_sentences_detailed_metadata

    def generate_multiple_cells(self):
        all_cells = []
        for _ in range(self.num_cells):
            cell = self.generate_cell()
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

    def store_dataframe(self, df_all_cells):
        path_to_stimuli = os.path.join("..", "..", "stimuli")
        if not os.path.exists(path_to_stimuli):
            os.makedirs(path_to_stimuli)
        fname = os.path.join(path_to_stimuli, "greek_sentences.csv")
        df_all_cells.to_csv(fname)


# %%
# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate counterbalanced Greek sentence stimuli for SVO/VSO word order experiments."
    )
    parser.add_argument(
        "--num_cells",
        type=int,
        default=4,
        help="Number of experimental cells (default: 4)",
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
    args = parser.parse_args()

    experiment = Experiment(
        num_cells=args.num_cells,
        presentation_time=args.presentation_time,
        soa=args.soa,
        question_display_time=args.question_display_time,
        response_time=args.response_time,
    )

    print("Generating stimuli...")
    all_cells = experiment.generate_multiple_cells()
    experiment.print_summary(all_cells)

    # Create and inspect the DataFrame
    df_all_cells = experiment.create_dataframe(all_cells)
    
    # Store the dataframe
    path_to_stimuli = args.output_dir
    if not os.path.exists(path_to_stimuli):
        os.makedirs(path_to_stimuli)
    fname = os.path.join(path_to_stimuli, args.output_filename)
    df_all_cells.to_csv(fname)
    print(f"\nStimuli saved to: {fname}")


if __name__ == "__main__":
    main()
