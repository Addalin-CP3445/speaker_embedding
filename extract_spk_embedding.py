#!/usr/bin/env python3
"""
This script extracts speaker embeddings for each utterance using SpeechBrain's 
EncoderClassifier and writes the embeddings in Kaldi (ESPnet) archive (ark/scp) format.
Usage: python extract_spk_embed.py --wav_scp path/to/wav.scp --out_dir path/to/output
"""

import argparse
import os
import torchaudio
import numpy as np
import kaldiio
from speechbrain.inference.speaker import EncoderClassifier

def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings using SpeechBrain's EncoderClassifier "
                    "and save in ESPnet (Kaldi) format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav_scp", type=str, required=True, 
                        help="Path to the kaldi-style wav.scp file.")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Output directory to save the ark and scp files.")
    parser.add_argument("--savedir", type=str, default="pretrained_models/spkrec-xvect-voxceleb",
                        help="Directory to save / load the pretrained model from SpeechBrain.")
    parser.add_argument("--source", type=str, default="speechbrain/spkrec-xvect-voxceleb",
                        help="The Hugging Face model source for the speaker recognition model.")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use for inference (e.g., cuda:0 or cpu).")
    return parser

def load_wav_scp(wav_scp_path):
    """
    Reads a kaldi-style wav.scp file and returns a dictionary mapping utterance IDs to file paths.
    """
    utt_dict = {}
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt_id = parts[0]
                wav_path = parts[1]
                utt_dict[utt_id] = wav_path
    return utt_dict

def main(args):
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Set up output ark and scp file paths
    ark_path = os.path.join(args.out_dir, "spk_embed.ark")
    scp_path = os.path.join(args.out_dir, "spk_embed.scp")
    
    # Load the pretrained SpeechBrain encoder classifier
    classifier = EncoderClassifier.from_hparams(
        source=args.source, 
        savedir=args.savedir,
        run_opts={"device": args.device}
    )
    
    # Load wav.scp into a dictionary mapping utter
