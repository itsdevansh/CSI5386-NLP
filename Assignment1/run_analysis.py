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
        self.words_only = self.filter_words()
        self.content_words = self.remove_stopwords()
        self.bigrams = self.compute_bigrams()

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
        """Computes total content words and lexical density (words without stopwords)."""
        total_content_words = len(self.content_words)
        unique_content_words = len(set(self.content_words))
        lexical_density = unique_content_words / total_content_words if total_content_words > 0 else 0
        return total_content_words, unique_content_words, lexical_density
    
    def compute_bigrams(self):
        """Computes frequency of bigrams from a list of words."""
        return Counter(bigrams(self.content_words)).most_common()
    
    def print_summary(self):
        """Prints key statistics."""
        total_tokens, unique_tokens, type_token_ratio = self.count_tokens()
        total_words, unique_words, type_token_ratio_words = self.compute_word_statistics()
        total_content_words, unique_content_words, lexical_density = self.compute_content_word_statistics()
        top_20_words = Counter(self.words_only).most_common(20)
        top_20_content_words = Counter(self.content_words).most_common(20)
        top_20_bigrams = self.bigrams[:20]
        tokens_once = self.count_single_occurrence_tokens()

        print(f"Total Tokens: {total_tokens:,d}")
        print(f"Unique Tokens: {unique_tokens:,d}")
        print(f"Type/Token Ratio: {type_token_ratio:.6f}")
        print(f"Tokens Appearing Only Once: {tokens_once:,d}")

        print(f"Total Words (Excluding Punctuation): {total_words:,d}")
        print(f"Unique Words: {unique_words:,d}")
        print(f"Type/Token Ratio (Words Only): {type_token_ratio_words:.6f}")

        print("\nTop 20 Most Frequent Words:")
        for word, freq in top_20_words:
            print(f"{word}: {freq}")

        print(f"\nTotal Content Words (Excluding Stopwords): {total_content_words:,d}")
        print(f"Unique Content Words: {unique_content_words:,d}")
        print(f"Lexical Density (Words Without Stopwords): {lexical_density:.6f}")

        print("\nTop 20 Most Frequent Content Words:")
        for word, freq in top_20_content_words:
            print(f"{word}: {freq}")

        print("\nTop 20 Most Frequent Bigrams:")
        for bigram, freq in top_20_bigrams:
            print(f"{bigram[0]} {bigram[1]}: {freq}")


    def run_analysis(self):
        """Runs all steps of the analysis and saves results."""
        self.print_summary()
        print("\nAnalysis complete! All results saved to files.")


if __name__ == "__main__":
    analyzer = CorpusAnalyzer("output.txt")
    analyzer.run_analysis()
