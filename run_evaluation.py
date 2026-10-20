"""
Evaluation script - RAG systems ablation study

Runs the 5 RAG systems on the 35 validated evaluation questions.
Total: 175 answers (5 systems × 35 questions)

Output: data/evaluation/results.jsonl (questions in data/evaluation/questions.jsonl)
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Import systems
from src.systems.system1_bm25_only import System1_BM25Only
from src.systems.system2_vector_only import System2_VectorOnly
from src.systems.system3_hybrid import System3_Hybrid
from src.systems.system4_hybrid_graph import System4_HybridGraph
from src.systems.system5_agentic import System5_Agentic


class EvaluationRunner:
    """Runs the full evaluation of the 5 systems."""

    def __init__(self, questions_path: str, output_path: str):
        self.questions_path = Path(questions_path)
        self.output_path = Path(output_path)
        self.questions = self._load_questions()

        # Paths to data
        self.project_root = Path(__file__).parent
        self.bm25_path = self.project_root / "data/indexes/bm25_index"
        self.faiss_path = self.project_root / "data/indexes/faiss_index"
        self.chunks_path = self.project_root / "data/chunks/chunks.jsonl"
        self.graph_path = self.project_root / "data/knowledge_graph/knowledge_graph.pkl"

        # Common parameters
        self.temperature = 0.0
        self.top_k = 8

        # Systems are initialized lazily to save memory
        self.systems = {
            1: None,  # System1_BM25Only
            2: None,  # System2_VectorOnly
            3: None,  # System3_Hybrid
            4: None,  # System4_HybridGraph
            5: None,  # System5_Agentic
        }

        print(f"\n{'='*80}")
        print(f"EVALUATION RUNNER - Ablation Study")
        print(f"{'='*80}")
        print(f"Questions: {len(self.questions)}")
        print(f"Systems: 5")
        print(f"Total evaluations: {len(self.questions) * 5}")
        print(f"Output: {output_path}")
        print(f"{'='*80}\n")

    def _load_questions(self) -> List[Dict]:
        """Loads the questions from JSONL."""
        questions = []
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line.strip()))
        return questions

    def _get_system(self, system_id: int):
        """Lazy loading of the systems (only when needed)."""
        if self.systems[system_id] is None:
            print(f"  Initializing System {system_id}...")

            if system_id == 1:
                self.systems[1] = System1_BM25Only(
                    bm25_index_path=self.bm25_path,
                    chunks_file=self.chunks_path,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            elif system_id == 2:
                self.systems[2] = System2_VectorOnly(
                    faiss_index_path=self.faiss_path,
                    chunks_file=self.chunks_path,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            elif system_id == 3:
                self.systems[3] = System3_Hybrid(
                    chunks_file=self.chunks_path,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            elif system_id == 4:
                self.systems[4] = System4_HybridGraph(
                    chunks_file=self.chunks_path,
                    graph_file=self.graph_path,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            elif system_id == 5:
                self.systems[5] = System5_Agentic()

            print(f"  System {system_id} ready\n")

        return self.systems[system_id]

    def _evaluate_single(self, system_id: int, question: Dict) -> Dict:
        """Evaluates one question on one system."""
        system = self._get_system(system_id)

        try:
            # Execute the query
            start_time = time.time()
            result = system.answer(question['question'])
            end_time = time.time()

            # Extract fields, depending on the system
            answer = result.get('answer', '')
            time_seconds = result.get('time_seconds', end_time - start_time)

            # System 5 has additional fields
            tool_calls = result.get('tool_calls', 0) if system_id == 5 else 0
            failed_tools = result.get('failed_tools', []) if system_id == 5 else []
            searches = result.get('searches', 0) if system_id == 5 else 0

            return {
                'question_id': question['id'],
                'system_id': system_id,
                'category': question['category'],
                'question': question['question'],
                'answer': answer,
                'expected_answer': question.get('expected_answer', ''),
                'retrieved_contexts': result.get('chunk_texts', []),  # used by RAGAS
                'time_seconds': round(time_seconds, 2),
                'tool_calls': tool_calls,
                'failed_tools': failed_tools,
                'searches': searches,
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'question_id': question['id'],
                'system_id': system_id,
                'category': question['category'],
                'question': question['question'],
                'answer': '',
                'expected_answer': question.get('expected_answer', ''),
                'time_seconds': 0,
                'tool_calls': 0,
                'failed_tools': [],
                'searches': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_full_evaluation(self):
        """Runs the full evaluation: 5 systems × all questions."""
        results = []
        total = len(self.questions) * 5
        completed = 0

        print(f"Starting full evaluation...\n")

        for q_idx, question in enumerate(self.questions, 1):
            print(f"\n{'-'*80}")
            print(f"Question {q_idx}/{len(self.questions)}: {question['question'][:60]}...")
            print(f"   Category: {question['category']} | Difficulty: {question.get('difficulty', 'N/A')}")
            print(f"{'-'*80}")

            for system_id in range(1, 6):
                print(f"\n  System {system_id}: ", end='', flush=True)

                result = self._evaluate_single(system_id, question)
                results.append(result)
                completed += 1

                # Progress indicator
                if result['success']:
                    answer = result.get('answer') or ''
                    answer_preview = answer[:50] + '...' if len(answer) > 50 else answer
                    print(f"OK {result['time_seconds']}s - {answer_preview}")
                else:
                    print(f"Error: {result['error']}")

                # Save incrementally (do not lose progress on a crash)
                self._save_results(results)

                # Progress summary
                progress_pct = (completed / total) * 100
                print(f"     Progress: {completed}/{total} ({progress_pct:.1f}%)")

        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total answers: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['success'])}")
        print(f"Errors: {sum(1 for r in results if not r['success'])}")
        print(f"Output: {self.output_path}")
        print(f"{'='*80}\n")

        return results

    def _save_results(self, results: List[Dict]):
        """Saves the results as JSONL (incremental)."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def run_system_subset(self, system_ids: List[int]):
        """Runs only some of the systems (for quick tests)."""
        results = []

        print(f"Running systems: {system_ids}\n")

        for q_idx, question in enumerate(self.questions, 1):
            print(f"\nQuestion {q_idx}/{len(self.questions)}: {question['question'][:60]}...")

            for system_id in system_ids:
                print(f"  System {system_id}: ", end='', flush=True)
                result = self._evaluate_single(system_id, question)
                results.append(result)

                if result['success']:
                    print(f"OK {result['time_seconds']}s")
                else:
                    print(f"Error: {result['error']}")

        self._save_results(results)
        return results


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run RAG systems evaluation')
    parser.add_argument('--questions', default='data/evaluation/questions.jsonl',
                        help='Path to questions file')
    parser.add_argument('--output', default='data/evaluation/results.jsonl',
                        help='Path to output results')
    parser.add_argument('--systems', nargs='+', type=int, choices=[1,2,3,4,5],
                        help='Run only specific systems (e.g., --systems 1 3 5)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run only first 3 questions (quick test)')

    args = parser.parse_args()

    runner = EvaluationRunner(args.questions, args.output)

    # Quick test mode: only the first 3 questions
    if args.quick_test:
        print("QUICK TEST MODE: first 3 questions only\n")
        runner.questions = runner.questions[:3]

    if args.systems:
        # Subset of systems
        runner.run_system_subset(args.systems)
    else:
        # Full evaluation
        runner.run_full_evaluation()

    print("\nEvaluation completed successfully")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
