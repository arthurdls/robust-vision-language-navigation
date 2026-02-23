import numpy as np
import logging
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming these exist in your project structure
from utils.base_llm_providers import BaseLLM, LLMFactory

logger = logging.getLogger(__name__)

class LofreeConformalPredictor:
    """
    Implements Conformal Prediction for Black-Box LLMs (No Logits).

    Based on:
    1. LofreeCP: Uses Frequency, Normalized Entropy, and Semantic Similarity.
    2. ConU: Uses semantic clustering and strict correctness coverage.
    """

    def __init__(self, llm_provider: BaseLLM, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            llm_provider: Instance of BaseLLM (e.g., OpenAIProvider).
            embedding_model: HuggingFace model for semantic similarity (default is lightweight).
        """
        self.llm = llm_provider
        # Load a lightweight encoder for Semantic Similarity (SS) calculations
        try:
            self.encoder = SentenceTransformer(embedding_model)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        self.q_hat = float('inf') # The calibrated threshold
        self.lambdas = (0.5, 0.5) # Default hyperparameters for Eq. 8 in LofreeCP

    def _normalize_text(self, text: str) -> str:
        """Simple normalization for exact matching during frequency counting."""
        return text.strip().lower()

    def _sample_responses(self, prompt: str, m: int, image: str = None) -> list[str]:
        """
        Samples 'm' responses from the LLM.
        """
        strict_prompt = (
            f"Question: {prompt}\n"
            "Constraint: Answer with the exact entity, number, or name only. "
            "No punctuation, no filler words, no full sentences.\n"
            "Answer:"
        )

        responses = []
        # In a real scenario, asyncio would be preferred here for speed.
        for _ in range(m):
            try:
                # High temperature to ensure diversity for Uncertainty Quantification (UQ)
                if image is None:
                    resp = self.llm.make_text_request(strict_prompt, temperature=1.0)
                else:
                    resp = self.llm.make_text_and_image_request(strict_prompt, image, temperature=1.0)
                responses.append(resp)
            except Exception as e:
                logger.warning(f"Sampling failed: {e}")
        return responses

    def _compute_entropy(self, frequencies: list[float], m: int) -> float:
        """
        Calculates Normalized Entropy (NE) as a prompt-wise uncertainty notion.
        Formula: H(x) = - sum(p * log(p)) / log(m)
        """
        if m <= 1: return 0.0
        probs = np.array(frequencies)
        # Avoid log(0) with small epsilon
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        normalized_entropy = entropy / np.log(m)
        return normalized_entropy

    def _calculate_nonconformity_scores(self, prompt: str, samples: list[str], 
                                        ground_truth: str = None, lambda1: float = 0.5, lambda2: float = 0.5, 
                                        image: str = None):
        """
        Calculates nonconformity scores based on LofreeCP Eq (8).
        N = -Frequency + lambda1 * Entropy - lambda2 * SemanticSimilarity
        """
        m = len(samples)
        if m == 0: return {}, 0.0

        # 1. Frequency Analysis (Coarse-grained uncertainty)
        counts = Counter([self._normalize_text(s) for s in samples])
        unique_responses = list(counts.keys())
        
        # Map normalized text back to one representative raw text (for embedding)
        norm_to_raw = {self._normalize_text(s): s for s in samples}
        
        frequencies = {k: v/m for k, v in counts.items()}
        
        # 2. Normalized Entropy (Prompt-wise fine-grained uncertainty)
        ne = self._compute_entropy(list(frequencies.values()), m)

        # 3. Semantic Similarity (Response-wise fine-grained uncertainty)
        # Use Weighted Centroid instead of Top-1
        
        # Encode all unique responses
        raw_candidates = [norm_to_raw[k] for k in unique_responses]
        embeddings = self.encoder.encode(raw_candidates) # Shape: (N_unique, Dim)
        
        # Calculate Weighted Centroid
        # We weight the embedding of each unique response by its frequency
        weights = np.array([frequencies[k] for k in unique_responses])
        weights = weights / np.sum(weights) # Ensure sum is 1
        
        # Weighted average of embeddings to find the "center of gravity" of the answers
        centroid = np.average(embeddings, axis=0, weights=weights).reshape(1, -1)
        
        # Calculate cosine similarity between all candidates and the CENTROID
        sims = cosine_similarity(embeddings, centroid).flatten()
        ss_map = {k: s for k, s in zip(unique_responses, sims)}

        # 4. Calculate N scores
        # Score = -Freq + L1*NE - L2*SS
        n_scores = {}
        for resp_norm in unique_responses:
            freq = frequencies[resp_norm]
            ss = ss_map[resp_norm]
            
            # LofreeCP Eq 8 adaptation
            score = -freq + (lambda1 * ne) - (lambda2 * ss)
            n_scores[resp_norm] = score

        # If calibration: we need the score of the Ground Truth (GT)
        gt_score = float('inf')
        if ground_truth:
            gt_norm = self._normalize_text(ground_truth)
            
            # Check if exact normalized GT is in our samples
            if gt_norm in n_scores:
                gt_score = n_scores[gt_norm]
            else:
                # Advanced: Check semantic similarity if exact match fails
                gt_emb = self.encoder.encode([ground_truth])
                # Compare GT to the centroid of the generated samples
                # If GT is close to the centroid, it means the model "knows" it roughly
                gt_sim_centroid = cosine_similarity(gt_emb, centroid).flatten()[0]
                
                # We can also check closest neighbor in samples to find a proxy score
                gt_sims_samples = cosine_similarity(embeddings, gt_emb).flatten()
                best_idx = np.argmax(gt_sims_samples)
                
                if gt_sims_samples[best_idx] > 0.85: # Strict threshold for semantic equivalence
                    best_match_key = unique_responses[best_idx]
                    gt_score = n_scores[best_match_key]
                else:
                    # If GT is totally different from samples, score is infinity (miscoverage)
                    gt_score = float('inf')

        return n_scores, gt_score

    def calibrate(self, calibration_data: list[tuple], alpha: float = 0.1, 
                  m_samples: int = 10, lambda1: float = 0.5, lambda2: float = 0.5):
        """
        Step 1: Calibration
        Calculates 'q_hat' such that the prediction set contains the true answer with prob 1 - alpha.
        
        Args:
            calibration_data: List of tuples. Can be (prompt, ground_truth) or (prompt, ground_truth, image_path).
            alpha: User-specified error rate (e.g., 0.1 for 90% coverage).
            m_samples: Number of samples to generate per question.
        """
        self.lambdas = (lambda1, lambda2)
        calibration_scores = []
        
        print(f"Starting calibration on {len(calibration_data)} items with alpha={alpha}...")
        
        for i, item in enumerate(calibration_data):
            # Handle optional image in tuple
            if len(item) == 3:
                x, y_true, img = item
            else:
                x, y_true = item
                img = None

            # 1. Sample M responses
            samples = self._sample_responses(x, m_samples, image=img)
            
            # 2. Calculate Nonconformity Score for the True Label
            _, gt_score = self._calculate_nonconformity_scores(
                x, samples, y_true, lambda1, lambda2, image=img
            )
            
            calibration_scores.append(gt_score)
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(calibration_data)} calibration pairs.")

        # 3. Compute Quantile
        # q_hat = Ceiling((n+1)(1-alpha)) / n -th largest value
        n = len(calibration_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        
        # We clamp the level to [0, 1] for numpy
        q_level = min(max(q_level, 0.0), 1.0)
        
        # Handling infinity (miscoverage in calibration)
        clean_scores = np.array(calibration_scores)
        self.q_hat = np.quantile(clean_scores, q_level, method='higher')
        
        print(f"Calibration complete. q_hat = {self.q_hat:.4f}")
        return self.q_hat

    def predict(self, prompt: str, image: str = None, m_samples: int = 10) -> dict:
        """
        Step 2: Prediction / Validation
        Generates a prediction set for a new prompt.
        
        Args:
            prompt: Text prompt.
            image: Optional path to image (for Vision-Language Models).
            m_samples: Number of samples to generate.
            
        Returns:
            Dict containing prediction_set, scores, and coverage metadata.
        """
        if self.q_hat == float('inf'):
            logger.warning("Model is uncalibrated or q_hat is infinite.")
        
        # 1. Sample
        samples = self._sample_responses(prompt, m_samples, image=image)
        
        # 2. Calculate Scores for candidates
        n_scores, _ = self._calculate_nonconformity_scores(
            prompt, samples, None, self.lambdas[0], self.lambdas[1], image=image
        )
        
        # 3. Construct Prediction Set
        # Include y if N(x, y) <= q_hat
        prediction_set = []
        for resp, score in n_scores.items():
            if score <= self.q_hat:
                # We return the normalized key; could also return the raw string if tracked
                prediction_set.append(resp)
                
        return {
            "prediction_set": prediction_set,
            "all_candidates": list(n_scores.keys()),
            "scores": n_scores,
            "q_threshold": self.q_hat
        }

# --- Example Usage ---

if __name__ == "__main__":
    # Setup LLM
    try:
        # Ensure your utils folder has this Factory and Provider
        llm = LLMFactory.create(provider_name="openai", model="gpt-4o-mini")
    except Exception as e:
        print(f"Provider initialization failed: {e}")
        exit()

    # Initialize Conformal Predictor
    cp = LofreeConformalPredictor(llm_provider=llm)

    # Dummy Calibration Data
    # Format: (Question, Exact Answer) OR (Question, Exact Answer, ImagePath)
    calibration_data = [
        # Geography
        ("What is the capital city of France?", "paris"),
        ("Which continent is the Sahara Desert located in?", "africa"),
        ("What is the largest country in the world by land area?", "russia"),
        ("What is the capital of Japan?", "tokyo"),
        ("Which US state is known as the Sunshine State?", "florida"),

        # Science & Math
        ("What is the chemical symbol for Gold?", "au"),
        ("What is the hardest natural mineral?", "diamond"),
        ("What is 15 multiplied by 4?", "60"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("Which planet is known as the Red Planet?", "mars"),

        # General Knowledge
        ("Who wrote the play 'Hamlet'?", "william shakespeare"),
        ("What is the primary color mixed with red to make purple?", "blue"),
        ("In which year did the Titanic sink?", "1912"),
        ("What is the opposite of 'hot'?", "cold"),
        ("How many legs does a spider have?", "8"),
    ]

    # Run Calibration
    # alpha=0.2 implies we want the set to contain the true answer 80% of the time.
    cp.calibrate(calibration_data, alpha=0.2, m_samples=5)

    # Run Prediction
    test_question = "Who painted the Mona Lisa?"
    result = cp.predict(test_question, m_samples=5)

    print(f"\nQuestion: {test_question}")
    print(f"Prediction Set (Size {len(result['prediction_set'])}): {result['prediction_set']}")

"""
1. Sample: You ask the LLM the same question M times (e.g., 10 times) with high temperature to get diverse answers.
2. Cluster: You group identical answers and count their frequencies.
3. Embed: You turn the text answers into vectors to see how similar they are to each other (Semantic Similarity).
4. Score: You calculate a "Non-Conformity Score" for every answer.
    Low Score = The answer is very standard, frequent, and semantically central.
    High Score = The answer is rare, weird, or an outlier.
5. Filter:
    In Calibration: You find the "cutoff score" (q_hat) that captures the correct answer 90% of the time (or based on what your alpha is).
    In Prediction: You keep all answers that are below that cutoff score.
"""