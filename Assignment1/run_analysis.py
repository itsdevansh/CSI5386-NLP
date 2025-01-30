from collections import Counter
import string
import nltk
from nltk.util import bigrams
from nltk.corpus import stopwords

# Download stopwords (if not already available)
nltk.download("stopwords")

class CorpusAnalyzer:
    def __init__(self, tokenized_file):
        """Initialize with the path to the tokenized output file."""
        self.tokenized_file = tokenized_file
        self.tokens = self.read_tokens()
        self.token_counts = Counter(self.tokens)
        self.sorted_tokens = self.token_counts.most_common()

    # 1. Read Tokenized File
    def read_tokens(self):
        """Reads tokenized output and returns a list of tokens."""
        with open(self.tokenized_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]  # Remove empty lines

    # 2. Compute Token Frequencies

    def count_tokens(self):
        total_tokens = len(self.tokens)
        unique_tokens = len(set(self.tokens))
        type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        return total_tokens, unique_tokens, type_token_ratio

    # 3. Write Token Frequencies to File
    def write_token_frequencies(self, sorted_tokens, output_file):
        """Writes token frequencies to a file."""
        with open(output_file, "w", encoding="utf-8") as f:
            for token, freq in sorted_tokens:
                f.write(f"{token}\t{freq}\n")

    
    def print_summary(self):
        """Prints key statistics."""
        total_tokens, unique_tokens, type_token_ratio = self.count_tokens()

        print(f"Total Tokens: {total_tokens:,d}")
        print(f"Unique Tokens: {unique_tokens:,d}")
        print(f"Type/Token Ratio: {type_token_ratio:.6f}")


    def run_analysis(self):
        """Runs all steps of the analysis and saves results."""
        self.print_summary()
        print("\nAnalysis complete! All results saved to files.")


if __name__ == "__main__":
    analyzer = CorpusAnalyzer("output.txt")
    analyzer.run_analysis()
