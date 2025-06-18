from rag_setup import *

# =====================================
# MODULE 7: QA CHAIN CREATION
# =====================================
def create_qa_chain(llm, vectorstore, prompt_template, config):
    """Create the QA chain with custom prompt"""
    logger.info("🔗 Creating Retrieval QA chain...")
    start_time = time.time()
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.RETRIEVAL_K}
            ),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        setup_time = time.time() - start_time
        logger.info(f"✅ QA chain created successfully in {setup_time:.2f} seconds")
        logger.info(f"🔍 Retrieval documents: {config.RETRIEVAL_K}")
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"❌ Error creating QA chain: {str(e)}")
        raise

def is_gnidp_related(question, keywords):
    """Check if question is related to GNIDP topics"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in keywords)

# =====================================
# MODULE 8: QUERY INTERFACE AND TESTING
# =====================================
class GNIDPQuerySystem:
    def __init__(self, qa_chain, config):
        self.qa_chain = qa_chain
        self.config = config
        self.query_count = 0
        self.total_response_time = 0
        self.cache = QueryCache(
            cache_dir=config.CACHE_DIR,
            ttl=config.CACHE_TTL
        ) if config.ENABLE_CACHE else None
        self.cache_hits = 0
        self.cache_misses = 0

    def query(self, question: str, format_output: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        self.query_count += 1
        
        if format_output:
            print(f"\n{'='*60}")
            print(f"🔍 QUERY #{self.query_count}: {question}")
            print(f"{'='*60}")
        
        if self.config.ENABLE_CACHE:
            cached_result = self.cache.get(question)
            if cached_result:
                self.cache_hits += 1
                response_time = time.time() - start_time
                if format_output:
                    print(f"🚀 Cache hit! Response time: {response_time:.2f}s")
                return cached_result

        if self.config.ENABLE_CACHE:
            self.cache_misses += 1
        
        if not is_gnidp_related(question, self.config.GNIDP_KEYWORDS):
            response_time = time.time() - start_time
            result = {
                "answer": "I can only answer questions related to Little Andaman only. Please ask a question about these subjects.",
                "response_time": response_time,
                "relevant": False,
                "query_number": self.query_count,
                "cached": False
            }
            
            if format_output:
                print(f"❌ Not GNIDP-related")
                print(f"🤖 Response: {result['answer']}")
                print(f"⏱️  Response time: {response_time:.2f}s")

            if self.config.ENABLE_CACHE:
                self.cache.set(question, result)
            
            return result
        
        try:
            if format_output:
                print(f"🔍 Searching vector database...")
            qa_result = self.qa_chain.invoke({"query": question})
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            cleaned_answer = clean_response(qa_result["result"])
            
            result = {
                "answer": cleaned_answer,
                "response_time": response_time,
                "relevant": True,
                "query_number": self.query_count,
                "cached": False
            }
            
            if self.config.ENABLE_CACHE:
                self.cache.set(question, result)
            
            if format_output:
                print(f"✅ GNIDP-related query processed")
                print(f"\n🤖 ANSWER:")
                print("-" * 50)
                print(result["answer"])
                print("-" * 50)
                
                print(f"⏱️  Response time: {response_time:.2f}s")
                print(f"📊 Average response time: {self.total_response_time/self.query_count:.2f}s")
                
                if self.config.ENABLE_CACHE:
                    total_queries = self.cache_hits + self.cache_misses
                    hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
                    print(f"💾 Cache Stats - Hit Rate: {hit_rate:.1f}% ({self.cache_hits}/{total_queries})")
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_result = {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "response_time": response_time,
                "relevant": True,
                "error": str(e),
                "query_number": self.query_count,
                "cached": False
            }
            
            if format_output:
                print(f"❌ ERROR: {str(e)}")
                print(f"⏱️  Response time: {response_time:.2f}s")
            return error_result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        if not self.config.ENABLE_CACHE:
            return {"cache_enabled": False}
            
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "cache_enabled": True,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%",
            **self.cache.get_stats()
        }
        
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        print(f"\n🚀 BATCH PROCESSING {len(questions)} QUERIES")
        print("="*70)
        
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        successful_queries = [r for r in results if 'error' not in r]
        avg_time = sum(r['response_time'] for r in results) / len(results)
        
        print(f"\n📊 BATCH SUMMARY:")
        print(f"Total queries: {len(questions)}")
        print(f"Successful: {len(successful_queries)}")
        print(f"Average response time: {avg_time:.2f}s")
        
        return results

# =====================================
# MODULE 9: LAZY INITIALIZATION
# =====================================
_query_system_instance = None

def get_query_system():
    global _query_system_instance
    if _query_system_instance is None:
        print("🔗 Creating Retrieval QA chain...")
        qa_chain = create_qa_chain(llm, vectorstore, prompt_template, config)
        print("✅ QA chain is ready!")

        print("🚀 Initializing GNIDP Query System...")
        _query_system_instance = GNIDPQuerySystem(qa_chain, config)
        print("✅ Query system ready!")

        print(f"\n🎯 TESTING COMPLETE!")
        print(f"✅ System is fully operational and ready for use!")
    return _query_system_instance

# =====================================
# MODULE 10: UTILITY FUNCTIONS
# =====================================
def quick_query(question: str):
    """Quick query function with simplified display formatting"""
    query_system = get_query_system()
    result = query_system.query(question, format_output=False)
    
    try:
        answer = result["answer"]
        if answer.startswith('{"') and answer.endswith('}'):
            import json
            parsed = json.loads(answer)
            answer = parsed.get("answer", answer)
    except:
        answer = result["answer"]
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print("\nAnswer:")
    print("-"*50)
    print(answer)
    print("-"*50)

    return None

def system_status():
    print("\n🔍 SYSTEM STATUS CHECK")
    print("="*40)
    query_system = get_query_system()
    
    components = {
        "📄 Documents": len(documents) if documents is not None else "Using existing vector store",
        "🧠 Embeddings": "✅ Loaded" if 'embeddings' in globals() else "❌ Not loaded",
        "🗃️  Vector Store": "✅ Ready" if 'vectorstore' in globals() else "❌ Not ready",
        "🤖 LLM": "✅ Connected" if 'llm' in globals() else "❌ Not connected",
        "🔗 QA Chain": "✅ Ready"
    }
    
    for component, status in components.items():
        print(f"{component}: {status}")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {config.OLLAMA_MODEL}")
    print(f"   Vector Store: {config.VECTORSTORE_TYPE}")
    print(f"   Chunk Size: {config.CHUNK_SIZE}")
    print(f"   Retrieval K: {config.RETRIEVAL_K}")
    
    if query_system.query_count > 0:
        print(f"\n📊 Performance:")
        print(f"   Queries processed: {query_system.query_count}")
        print(f"   Average response time: {query_system.total_response_time/query_system.query_count:.2f}s")

def save_conversation(conversation_log: List[Dict], filename: str = None):
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gnidp_conversation_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("GNIDP RAG Chatbot Conversation Log\n")
        f.write("="*50 + "\n\n")
        
        for i, entry in enumerate(conversation_log, 1):
            f.write(f"Query {i}: {entry['question']}\n")
            f.write(f"Response: {entry['answer']}\n")
            f.write(f"Response Time: {entry['response_time']:.2f}s\n")
            f.write(f"Relevant: {entry['relevant']}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"💾 Conversation saved to {filename}")

# =====================================
# OPTIONAL: INITIAL QUICK TEST CALLS (can comment/remove)
# =====================================
# quick_query("What are the economic benefits of GNIDP?")
# quick_query("How much land?")
