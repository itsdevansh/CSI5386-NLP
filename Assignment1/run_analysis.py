from collections import Counter
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
        self.words_only = self.filter_words()
        self.content_words = self.remove_stopwords()
        self.bigrams = self.compute_bigrams()

    def read_tokens(self):
        """Reads tokenized output and returns a list of tokens."""
        with open(self.tokenized_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]  # Remove empty lines

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

    def compute_word_statistics(self):
        """Computes total words, unique words, and type/token ratio for words only."""
        total_words = len(self.words_only)
        unique_words = len(set(self.words_only))
        type_token_ratio_words = unique_words / total_words if total_words > 0 else 0
        return total_words, unique_words, type_token_ratio_words

    def remove_stopwords(self):
        """Removes stopwords from token list."""
        stop_words = set(stopwords.words("english"))
        return [token for token in self.words_only if token.lower() not in stop_words]

    def compute_content_word_statistics(self):
        """Computes total content words and lexical density."""
        total_content_words = len(self.content_words)
        unique_content_words = len(set(self.content_words))
        lexical_density = unique_content_words / total_content_words if total_content_words > 0 else 0
        return total_content_words, unique_content_words, lexical_density

    def compute_bigrams(self):
        """Computes frequency of bigrams from a list of words."""
        return Counter(bigrams(self.content_words)).most_common()

    def generate_report(self):
        """Generates a structured report and saves it to report.txt"""
        total_tokens, unique_tokens, type_token_ratio = self.count_tokens()
        total_words, unique_words, type_token_ratio_words = self.compute_word_statistics()
        total_content_words, unique_content_words, lexical_density = self.compute_content_word_statistics()
        tokens_once = self.count_single_occurrence_tokens()
        top_20_words = Counter(self.words_only).most_common(20)
        top_20_content_words = Counter(self.content_words).most_common(20)
        top_20_bigrams = self.bigrams[:20]

        with open("report.txt", "w", encoding="utf-8") as report:
            # Section (a): First 20 lines of output.txt
            report.write("### (a) First 20 lines of Tokenized Output (output.txt):\n")
            try:
                with open(self.tokenized_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f][:20]
                    report.write("\n".join(lines) + "\n\n")
            except FileNotFoundError:
                report.write("ERROR: output.txt not found.\n\n")

            # Section (b): Token and Type Statistics
            report.write("### (b) Token and Type Statistics:\n")
            report.write(f"Total Tokens: {total_tokens:,d}\n")
            report.write(f"Unique Tokens: {unique_tokens:,d}\n")
            report.write(f"Type/Token Ratio: {type_token_ratio:.6f}\n\n")

            # Section (c): First 20 lines from tokens.txt
            report.write("### (c) First 20 Lines from Token Frequency File (tokens.txt):\n")
            try:
                with open("tokens.txt", "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f][:20]
                    report.write("\n".join(lines) + "\n\n")
            except FileNotFoundError:
                report.write("ERROR: tokens.txt not found.\n\n")

            # Section (d): Tokens Appearing Only Once
            report.write("### (d) Tokens Appearing Only Once:\n")
            report.write(f"{tokens_once:,d} tokens appear only once in the corpus.\n\n")

            # Section (e): Word Extraction and Lexical Diversity
            report.write("### (e) Word Count and Lexical Diversity:\n")
            report.write(f"Total Words (Excluding Punctuation): {total_words:,d}\n")
            report.write(f"Unique Words: {unique_words:,d}\n")
            report.write(f"Type/Token Ratio (Words Only): {type_token_ratio_words:.6f}\n")
            report.write("\nTop 20 Most Frequent Words:\n")
            for word, freq in top_20_words:
                report.write(f"{word}: {freq}\n")
            report.write("\n")

            # Section (f): Stopwords Removal and Lexical Density
            report.write("### (f) Content Words and Lexical Density:\n")
            report.write(f"Total Content Words (Excluding Stopwords): {total_content_words:,d}\n")
            report.write(f"Unique Content Words: {unique_content_words:,d}\n")
            report.write(f"Lexical Density: {lexical_density:.6f}\n")
            report.write("\nTop 20 Most Frequent Content Words:\n")
            for word, freq in top_20_content_words:
                report.write(f"{word}: {freq}\n")
            report.write("\n")

            # Section (g): Most Frequent Bigrams
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
    analyzer = CorpusAnalyzer("output_nltk.txt")
    analyzer.run_analysis()
