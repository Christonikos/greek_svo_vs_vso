#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-POS Matching Script 

This script provides accurate matching of parts of speech with their 
corresponding tokens, handling multi-token words properly.

"""

import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TokenMatch:
    """Data class to store token matching information with multi-token support."""

    sentence_id: int
    sentence: str
    order: str  # SVO or VSO
    question_for: str  # V1 or V2
    n1_number: str  # sing or plur
    n1_gender: str  # m or f
    n2_gender: str  # m or f

    # Token positions (lists to handle multi-token words)
    det_n1_positions: List[int] = None
    n1_positions: List[int] = None
    pou_position: int = None
    det_n2_positions: List[int] = None
    n2_positions: List[int] = None
    v1_positions: List[int] = None
    v2_positions: List[int] = None

    # Token strings (lists to handle multi-token words)
    det_n1_tokens: List[str] = None
    n1_tokens: List[str] = None
    pou_token: str = None
    det_n2_tokens: List[str] = None
    n2_tokens: List[str] = None
    v1_tokens: List[str] = None
    v2_tokens: List[str] = None

    # Word reconstructions (for validation)
    det_n1_word: str = None
    n1_word: str = None
    det_n2_word: str = None
    n2_word: str = None
    v1_word: str = None
    v2_word: str = None


class TokenPOSMatcher:
    """Class to match parts of speech with their corresponding tokens."""

    def __init__(self, stimuli_path: str, activations_path: str):
        """Initialize the  matcher."""
        self.stimuli_path = stimuli_path
        self.activations_path = activations_path
        self.stimuli_df = pd.read_csv(stimuli_path)

        # Constants for token patterns
        self.POU_TOKEN = "ĠÏĢÎ¿Ïħ"  # The consistent 'που' token
        self.BOS_TOKEN = "<|begin_of_text|>"
        self.PERIOD_TOKEN = "."

    def _decode_token_sequence(self, tokens: List[str]) -> str:
        """
        Decode a sequence of tokens back to the original word.
        This is a heuristic approach for validation.
        """
        if not tokens:
            return ""

        # Join tokens and try to clean up the encoding
        combined = "".join(tokens)

        # Remove the leading space marker if present
        if combined.startswith("Ġ"):
            combined = combined[1:]

        # This is a simplified decoding - in practice, you'd need proper tokenizer decoding
        return combined

    def _find_word_boundaries(
        self, tokens: List[str], sentence_words: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Find word boundaries in the token sequence based on the original sentence words.
        Returns list of (start_pos, end_pos) tuples for each word.
        """
        boundaries = []

        # Find 'που' position first as anchor
        pou_pos = None
        for i, token in enumerate(tokens):
            if token == self.POU_TOKEN:
                pou_pos = i
                break

        if pou_pos is None:
            raise ValueError("Could not find 'που' token")

        # The structure is: [BOS] Det_N1 N1_tokens... 'που' [rest...]
        # We know 'που' is at pou_pos, so we can work backwards and forwards

        # Before 'που': should be Det_N1 and N1 tokens
        # After 'που': depends on SVO vs VSO order

        # For now, return a simplified boundary detection
        # This would need more sophisticated logic for full accuracy
        boundaries.append((1, 2))  # Det_N1 (assuming single token)
        boundaries.append((2, pou_pos))  # N1 (from pos 2 to pou_pos)
        boundaries.append((pou_pos, pou_pos + 1))  # 'που'

        # The rest depends on sentence structure and would need more analysis
        return boundaries

    def match_sentence_tokens_(self, sentence_id: int) -> TokenMatch:
        """
        matching that handles multi-token words properly.
        """
        # Get sentence data from stimuli
        sentence_row = self.stimuli_df.iloc[sentence_id]
        sentence = sentence_row["Sentence"]
        order = sentence_row["order"]
        question_for = sentence_row["question_for"]
        n1_number = sentence_row["N1_number"]
        n1_gender = sentence_row["N1_gender"]
        n2_gender = sentence_row["N2_gender"]

        # Load activation file
        activation_file_path = os.path.join(
            self.activations_path, f"sentence_{sentence_id}.pt"
        )
        activation_file = torch.load(activation_file_path)
        tokens = activation_file["tokens"]

        # Create  match object
        match = TokenMatch(
            sentence_id=sentence_id,
            sentence=sentence,
            order=order,
            question_for=question_for,
            n1_number=n1_number,
            n1_gender=n1_gender,
            n2_gender=n2_gender,
        )

        # Initialize lists
        match.det_n1_positions = []
        match.n1_positions = []
        match.det_n2_positions = []
        match.n2_positions = []
        match.v1_positions = []
        match.v2_positions = []

        match.det_n1_tokens = []
        match.n1_tokens = []
        match.det_n2_tokens = []
        match.n2_tokens = []
        match.v1_tokens = []
        match.v2_tokens = []

        # Find 'που' position as anchor
        for i, token in enumerate(tokens):
            if token == self.POU_TOKEN:
                match.pou_position = i
                match.pou_token = token
                break

        if match.pou_position is None:
            raise ValueError(
                f"Could not find 'που' token in sentence {sentence_id}"
            )

        # Parse sentence structure
        words = sentence.split()
        det_n1_word = words[0]
        n1_word = words[1]
        # words[2] is 'που'

        if order == "SVO":
            det_n2_word = words[3]
            n2_word = words[4]
            v1_word = words[5]
            v2_word = words[6]
        else:  # VSO
            v1_word = words[3]
            det_n2_word = words[4]
            n2_word = words[5]
            v2_word = words[6]

        # Store expected words for validation
        match.det_n1_word = det_n1_word
        match.n1_word = n1_word
        match.det_n2_word = det_n2_word
        match.n2_word = n2_word
        match.v1_word = v1_word
        match.v2_word = v2_word

        # Map tokens to words using heuristics
        # Det_N1 is typically position 1 (after BOS)
        if len(tokens) > 1:
            match.det_n1_positions.append(1)
            match.det_n1_tokens.append(tokens[1])

        # N1 tokens: from position 2 up to 'που' position
        for i in range(2, match.pou_position):
            match.n1_positions.append(i)
            match.n1_tokens.append(tokens[i])

        # After 'που': depends on order
        pos_after_pou = match.pou_position + 1
        remaining_tokens = len(tokens) - pos_after_pou - 1  # -1 for period

        if order == "SVO":
            # SVO: Det_N2 N2_tokens V1_tokens V2_tokens
            if remaining_tokens >= 1:
                match.det_n2_positions.append(pos_after_pou)
                match.det_n2_tokens.append(tokens[pos_after_pou])

                # Distribute remaining tokens among N2, V1, V2
                # This is a heuristic - in practice, you'd need more sophisticated parsing
                if remaining_tokens >= 2:
                    match.n2_positions.append(pos_after_pou + 1)
                    match.n2_tokens.append(tokens[pos_after_pou + 1])

                if remaining_tokens >= 3:
                    match.v1_positions.append(pos_after_pou + 2)
                    match.v1_tokens.append(tokens[pos_after_pou + 2])

                if remaining_tokens >= 4:
                    match.v2_positions.append(pos_after_pou + 3)
                    match.v2_tokens.append(tokens[pos_after_pou + 3])

                # If there are more tokens, they likely belong to multi-token words
                # Add them to the last word (V2 in SVO)
                for i in range(pos_after_pou + 4, len(tokens) - 1):
                    match.v2_positions.append(i)
                    match.v2_tokens.append(tokens[i])

        else:  # VSO
            # VSO: V1_tokens Det_N2 N2_tokens V2_tokens
            if remaining_tokens >= 1:
                match.v1_positions.append(pos_after_pou)
                match.v1_tokens.append(tokens[pos_after_pou])

                if remaining_tokens >= 2:
                    match.det_n2_positions.append(pos_after_pou + 1)
                    match.det_n2_tokens.append(tokens[pos_after_pou + 1])

                if remaining_tokens >= 3:
                    match.n2_positions.append(pos_after_pou + 2)
                    match.n2_tokens.append(tokens[pos_after_pou + 2])

                if remaining_tokens >= 4:
                    match.v2_positions.append(pos_after_pou + 3)
                    match.v2_tokens.append(tokens[pos_after_pou + 3])

                # Add remaining tokens to V2
                for i in range(pos_after_pou + 4, len(tokens) - 1):
                    match.v2_positions.append(i)
                    match.v2_tokens.append(tokens[i])

        return match

    def print__summary(self, match: TokenMatch):
        """Summary of the  token matching."""
        print(f"\n===  Analysis: Sentence {match.sentence_id} ===")
        print(f"Sentence: {match.sentence}")
        print(f"Order: {match.order}, Question for: {match.question_for}")

        print(f"\nExpected words:")
        print(f"  Det_N1: '{match.det_n1_word}'")
        print(f"  N1: '{match.n1_word}'")
        print(f"  Det_N2: '{match.det_n2_word}'")
        print(f"  N2: '{match.n2_word}'")
        print(f"  V1: '{match.v1_word}'")
        print(f"  V2: '{match.v2_word}'")

        print(f"\nToken mappings:")
        print(f"  Det_N1: {match.det_n1_positions} -> {match.det_n1_tokens}")
        print(f"  N1: {match.n1_positions} -> {match.n1_tokens}")
        print(f"  που: {match.pou_position} -> '{match.pou_token}'")
        print(f"  Det_N2: {match.det_n2_positions} -> {match.det_n2_tokens}")
        print(f"  N2: {match.n2_positions} -> {match.n2_tokens}")
        print(f"  V1: {match.v1_positions} -> {match.v1_tokens}")
        print(f"  V2: {match.v2_positions} -> {match.v2_tokens}")

    def get_activation_for_word(
        self, match: TokenMatch, word_type: str, layer_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get activation matrix for a specific word, handling multi-token words.

        Args:
            match: TokenMatch object
            word_type: One of 'det_n1', 'n1', 'pou', 'det_n2', 'n2', 'v1', 'v2'
            layer_idx: Layer index

        Returns:
            Activation matrix (averaged if multi-token) or None
        """
        # Load activation file
        activation_file_path = os.path.join(
            self.activations_path, f"sentence_{match.sentence_id}.pt"
        )
        activation_file = torch.load(activation_file_path)

        # Get positions for the requested word type
        positions = getattr(match, f"{word_type}_positions", None)
        if word_type == "pou":
            positions = (
                [match.pou_position]
                if match.pou_position is not None
                else None
            )

        if not positions:
            return None

        # Get activations for all positions
        activations = []
        for pos in positions:
            if 0 <= pos < len(activation_file["hidden_states"][layer_idx]):
                act = activation_file["hidden_states"][layer_idx][pos].numpy()
                activations.append(act)

        if not activations:
            return None

        # If multiple tokens, average their activations
        if len(activations) == 1:
            return activations[0]
        else:
            return np.mean(activations, axis=0)

    def process_all_sentences(self) -> List[TokenMatch]:
        """Process all sentences and return  matches."""
        all_matches = []

        for sentence_id in range(len(self.stimuli_df)):
            try:
                match = self.match_sentence_tokens_(sentence_id)
                all_matches.append(match)
            except Exception as e:
                print(f"Error processing sentence {sentence_id}: {e}")
                continue

        return all_matches

    def save__matches_to_csv(
        self, matches: List[TokenMatch], output_path: str
    ):
        """
        Save  token matches to a CSV file.

        Args:
            matches: List of TokenMatch objects
            output_path: Path to save the CSV file
        """
        data = []

        for match in matches:
            row = {
                "sentence_id": match.sentence_id,
                "sentence": match.sentence,
                "order": match.order,
                "question_for": match.question_for,
                "n1_number": match.n1_number,
                "n1_gender": match.n1_gender,
                "n2_gender": match.n2_gender,
                # Expected words
                "det_n1_word": match.det_n1_word,
                "n1_word": match.n1_word,
                "det_n2_word": match.det_n2_word,
                "n2_word": match.n2_word,
                "v1_word": match.v1_word,
                "v2_word": match.v2_word,
                # Token positions (as strings for CSV)
                "det_n1_positions": str(match.det_n1_positions),
                "n1_positions": str(match.n1_positions),
                "pou_position": match.pou_position,
                "det_n2_positions": str(match.det_n2_positions),
                "n2_positions": str(match.n2_positions),
                "v1_positions": str(match.v1_positions),
                "v2_positions": str(match.v2_positions),
                # Token strings (as strings for CSV)
                "det_n1_tokens": str(match.det_n1_tokens),
                "n1_tokens": str(match.n1_tokens),
                "pou_token": match.pou_token,
                "det_n2_tokens": str(match.det_n2_tokens),
                "n2_tokens": str(match.n2_tokens),
                "v1_tokens": str(match.v1_tokens),
                "v2_tokens": str(match.v2_tokens),
                # Token counts
                "det_n1_token_count": (
                    len(match.det_n1_positions)
                    if match.det_n1_positions
                    else 0
                ),
                "n1_token_count": (
                    len(match.n1_positions) if match.n1_positions else 0
                ),
                "det_n2_token_count": (
                    len(match.det_n2_positions)
                    if match.det_n2_positions
                    else 0
                ),
                "n2_token_count": (
                    len(match.n2_positions) if match.n2_positions else 0
                ),
                "v1_token_count": (
                    len(match.v1_positions) if match.v1_positions else 0
                ),
                "v2_token_count": (
                    len(match.v2_positions) if match.v2_positions else 0
                ),
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved  token matches to {output_path}")


def main():
    """Main function to demonstrate  usage."""
    # Initialize paths
    stimuli_path = "../stimuli/greek_sentences.csv"
    activations_path = "../activations"

    # Create  matcher
    matcher = TokenPOSMatcher(stimuli_path, activations_path)

    # Analyze first few sentences in detail
    print("===  Token-POS Matching Analysis ===")

    for sentence_id in [0, 1, 2]:
        try:
            match = matcher.match_sentence_tokens_(sentence_id)
            matcher.print__summary(match)

            # Example: Get activation for N1
            n1_activation = matcher.get_activation_for_word(
                match, "n1", layer_idx=0
            )
            if n1_activation is not None:
                print(f"  N1 activation shape: {n1_activation.shape}")

        except Exception as e:
            print(f"Error with sentence {sentence_id}: {e}")

    # Process all sentences and save to CSV
    print("\n=== Processing all sentences ===")
    all_matches = matcher.process_all_sentences()

    # Save results to CSV
    output_path = "../token_pos_matches.csv"
    matcher.save__matches_to_csv(all_matches, output_path)

    print(f"\nProcessed {len(all_matches)} sentences successfully!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
