import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams
from collections import Counter
import os

nltk.download("punkt")
nltk.download("stopwords")

class CorpusAnalyzer:
    def __init__(self, raw_text_file, tokenized_file):
        """Initialize with the path to the raw text file and tokenized output file."""
        self.raw_text_file = raw_text_file
        self.tokenized_file = tokenized_file
        
        # Step 1: Tokenize the raw text and save to file
        self.tokenize_text()
        
        # Step 2: Read the tokenized data for analysis
        self.tokens = self.read_tokens()
        self.token_counts = Counter(self.tokens)
        self.sorted_tokens = self.token_counts.most_common()
        self.words_only = self.filter_words()
        self.content_words = self.remove_stopwords()
        self.bigrams = self.compute_bigrams()

    def tokenize_text(self):
        """Tokenizes the raw text file and saves the output to a file."""
        if not os.path.exists(self.raw_text_file):
            print(f"ERROR: {self.raw_text_file} not found.")
            return

        with open(self.raw_text_file, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = word_tokenize(text)  # Tokenize the text using NLTK
        
        # Write the tokenized output to a file
        with open(self.tokenized_file, "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(token + "\n")

    def read_tokens(self):
        """Reads tokenized output and returns a list of tokens."""
        with open(self.tokenized_file, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]  # Remove empty lines

    def count_tokens(self):
        """Computes total token count, unique tokens, and type-token ratio."""
        total_tokens = len(self.tokens)
        unique_tokens = len(set(self.tokens))
        type_token_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
        return total_tokens, unique_tokens, type_token_ratio

    def write_token_frequencies(self):
        """Writes token frequencies to a file."""
        with open("tokens.txt", "w", encoding="utf-8") as f:
            for token, freq in self.sorted_tokens:
                f.write(f"{token}\t{freq}\n")

    def count_single_occurrence_tokens(self):
        """Counts the number of tokens that appear only once."""
        return sum(1 for token, freq in self.token_counts.items() if freq == 1)

    def filter_words(self):
        """Filters out punctuation and symbols, keeping only alphabetic words."""
        return [token for token in self.tokens if token.isalpha()]

    def remove_stopwords(self):
        """Removes stopwords from token list."""
        stop_words = set(stopwords.words("english"))
        return [token for token in self.words_only if token.lower() not in stop_words]

    def compute_bigrams(self):
        """Computes frequency of bigrams from a list of words."""
        return Counter(bigrams(self.content_words)).most_common()

    def generate_report(self):
        """Generates a structured report and saves it to report.txt"""
        total_tokens, unique_tokens, type_token_ratio = self.count_tokens()
        tokens_once = self.count_single_occurrence_tokens()
        top_20_words = Counter(self.words_only).most_common(20)
        top_20_content_words = Counter(self.content_words).most_common(20)
        top_20_bigrams = self.bigrams[:20]

        with open("report.txt", "w", encoding="utf-8") as report:
            report.write("### (a) First 20 Lines of Tokenized Output (output.txt):\n")
            with open(self.tokenized_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f][:20]
                report.write("\n".join(lines) + "\n\n")

            report.write("### (b) Token and Type Statistics:\n")
            report.write(f"Total Tokens: {total_tokens:,d}\n")
            report.write(f"Unique Tokens: {unique_tokens:,d}\n")
            report.write(f"Type/Token Ratio: {type_token_ratio:.6f}\n\n")

            report.write("### (c) First 20 Lines from Token Frequency File (tokens.txt):\n")
            with open("tokens.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f][:20]
                report.write("\n".join(lines) + "\n\n")

            report.write("### (d) Tokens Appearing Only Once:\n")
            report.write(f"{tokens_once:,d} tokens appear only once in the corpus.\n\n")

            report.write("### (e) Word Count and Lexical Diversity:\n")
            report.write(f"Total Words: {len(self.words_only):,d}\n")
            report.write(f"Type/Token Ratio (Words Only): {len(set(self.words_only)) / len(self.words_only) if self.words_only else 0:.6f}\n")
            report.write("Top 20 Most Frequent Words:\n")
            for word, freq in top_20_words:
                report.write(f"{word}: {freq}\n")
            report.write("\n")

            report.write("### (f) Content Words and Lexical Density:\n")
            report.write(f"Total Content Words: {len(self.content_words):,d}\n")
            report.write(f"Lexical Density: {len(set(self.content_words)) / len(self.content_words) if self.content_words else 0:.6f}\n")
            report.write("Top 20 Most Frequent Content Words:\n")
            for word, freq in top_20_content_words:
                report.write(f"{word}: {freq}\n")
            report.write("\n")

            report.write("### (g) Most Frequent Bigrams:\n")
            for bigram, freq in top_20_bigrams:
                report.write(f"{bigram[0]} {bigram[1]}: {freq}\n")
            report.write("\n")

        print("Analysis complete! Report saved to report.txt.")

    def run_analysis(self):
        """Runs all steps of the analysis and saves results."""
        self.write_token_frequencies()
        self.generate_report()

if __name__ == "__main__":
    analyzer = CorpusAnalyzer("concatenated_text.txt", "output.txt")
    analyzer.run_analysis()
