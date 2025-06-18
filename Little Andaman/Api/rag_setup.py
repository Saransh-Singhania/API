# =====================================
# MODULE 1: DEPENDENCIES AND IMPORTS
# =====================================
# Run this cell first to install and import all required dependencies

# Install dependencies (run once)
# !pip install langchain langchain-community chromadb pypdf sentence-transformers faiss-cpu numpy huggingface-hub requests torch

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json


# Core libraries
import chromadb
from chromadb.config import Settings
import faiss
import numpy as np

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅ All dependencies imported successfully!")
print("📁 Make sure your PDF files are in the './gnidp_pdfs/' directory")
print("🤖 Make sure Ollama is running with: ollama serve")
print("📦 Make sure gemma3:4b model is pulled: ollama pull gemma3:4b")

# Additional imports for caching
from diskcache import Cache
from functools import lru_cache
from typing import List, Dict, Any, Optional  # Add Optional to existing imports
import hashlib
import json

# Cache module
class QueryCache:
    """Handles multi-level caching for GNIDP RAG system"""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.memory_cache: Dict[str, Any] = {}
        self.cache_dir = cache_dir
        self.ttl = ttl  # Time-to-live in seconds
        os.makedirs(cache_dir, exist_ok=True)
        self.disk_cache = Cache(cache_dir)
        
    def _generate_cache_key(self, query: str) -> str:
        """Generate consistent cache key from query"""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _get_from_memory(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from memory cache with LRU strategy"""
        result = self.memory_cache.get(cache_key)
        if result and (time.time() - result['timestamp']) < self.ttl:
            return result['data']
        return None
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result using multi-level cache strategy"""
        cache_key = self._generate_cache_key(query)
        
        # Try memory cache first
        result = self._get_from_memory(cache_key)
        if result:
            return result
        
        # Try disk cache
        result = self.disk_cache.get(cache_key)
        if result:
            # Promote to memory cache
            self.set(query, result)
            return result
            
        return None
    
    def set(self, query: str, result: Dict[str, Any]) -> None:
        """Store result in both memory and disk cache"""
        cache_key = self._generate_cache_key(query)
        
        # Memory cache
        self.memory_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        # Disk cache
        self.disk_cache.set(cache_key, result, expire=self.ttl)
    
    def clear(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.disk_cache),
            'cache_dir': self.cache_dir,
            'ttl': self.ttl
        }
    
# =====================================
# MODULE 2: CONFIGURATION SETTINGS
# =====================================
# Configure all parameters for the RAG system

class GNIDPConfig:
    """Configuration class for GNIDP RAG Chatbot"""
    
    def __init__(self):
        # File paths
        self.PDF_DIRECTORY = r"C:\Users\ACER\Documents\NIC_intern\Little Andaman\Api\D_set"  # UPDATE THIS PATH TO YOUR PDF DIRECTORY
        self.VECTORSTORE_DIR = r"C:\Users\ACER\Documents\NIC_intern\Little Andaman\Api\V_set"  # Vector store persistence directory
        self.VECTORSTORE_FILENAME = "LA_vectorstore"  # Removed .faiss extension
        self.REBUILD_VECTORSTORE = False # Changed to True to force rebuild
        
        # Model settings
        self.OLLAMA_MODEL = "qwen3:0.6b"  # Ollama model for LLM
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Performance-optimized chunking parameters
        self.CHUNK_SIZE = 500  # Down from 1000
        self.CHUNK_OVERLAP = 20  # Reduced from 50
        
        # Vector store settings
        self.VECTORSTORE_TYPE = "faiss"  # "faiss" for speed, "chroma" for persistence
        self.RETRIEVAL_K = 1  # Reduced from 3 for faster retrieval
        
        # LLM parameters
        self.TEMPERATURE = 0.2
        self.TOP_K = 1
        self.TOP_P = 0.9
        self.NUM_CTX = 512  # Reduced context window for faster processing
        
        # GNIDP-specific keywords for filtering
        self.GNIDP_KEYWORDS = [
            # Island Names & Geography
    'little andaman', 'little andaman island', 'south andaman', 'andaman islands',
    'bay of bengal', 'andaman sea', 'ten degree channel', 'car nicobar',
    
    # Administrative & Locations
    'hut bay', 'dugong creek', 'butler bay', 'netaji nagar', 'harminder bay',
    'jackson creek', 'whisper wave', 'kala pathar', 'white surf', 'lalaji bay',
    'rampur', 'govind nagar', 'om shanti', 'khari nadi', 'bumila',
    
    # Beaches & Coastal Areas
    'butler bay beach', 'kala pathar beach', 'white surf beach', 'whisper wave beach',
    'netaji nagar beach', 'harminder bay beach', 'lalaji bay beach', 'om shanti beach',
    'beach', 'coastline', 'shore', 'bay', 'creek', 'surf point',
    
    # Indigenous Communities & Tribes
    'onge', 'onge tribe', 'indigenous people', 'tribal community', 'primitive tribe',
    'particularly vulnerable tribal group', 'pvtg', 'aboriginal', 'native population',
    'tribal rights', 'tribal land', 'forest rights', 'customary rights',
    
    # Wildlife & Biodiversity
    'elephant', 'asian elephant', 'wild elephant', 'elephant corridor',
    'saltwater crocodile', 'estuarine crocodile', 'sea turtle', 'olive ridley',
    'leatherback turtle', 'green turtle', 'hawksbill turtle', 'turtle nesting',
    'dugong', 'dolphins', 'whales', 'coral reef', 'mangroves', 'tropical forest',
    'endemic species', 'biodiversity', 'wildlife sanctuary', 'marine life',
    'bird watching', 'migratory birds', 'nesting sites', 'breeding grounds',
    
    # Flora & Forest
    'tropical rainforest', 'evergreen forest', 'littoral forest', 'mangrove forest',
    'padauk', 'gurjan', 'mahua', 'bamboo', 'rattan', 'medicinal plants',
    'timber', 'forest cover', 'deforestation', 'forest conservation',
    
    # Marine & Aquatic
    'coral reef', 'coral bleaching', 'marine ecosystem', 'seagrass beds',
    'fishing', 'marine protected area', 'coastal zone', 'tidal zone',
    'marine biodiversity', 'reef fish', 'shark', 'ray', 'barracuda',
    
    # Tourism & Activities
    'eco-tourism', 'beach tourism', 'adventure tourism', 'surfing', 'scuba diving',
    'snorkeling', 'water sports', 'jungle trekking', 'elephant safari',
    'nature walk', 'bird watching', 'fishing', 'boating', 'swimming',
    'tourist spots', 'tourist attractions', 'accommodation', 'resorts', 'hotels',
    
    # Transportation & Connectivity
    'helicopter', 'helicopter service', 'ship', 'ferry', 'boat service',
    'inter-island connectivity', 'port blair', 'hut bay jetty', 'landing ground',
    'road network', 'transportation', 'accessibility', 'remote location',
    
    # Development & Infrastructure
    'infrastructure development', 'road construction', 'jetty development',
    'tourism infrastructure', 'sustainable development', 'carrying capacity',
    'development pressure', 'urbanization', 'settlement', 'village development',
    
    # Environmental Concerns
    'environmental impact', 'climate change', 'sea level rise', 'coastal erosion',
    'tsunami', 'cyclone', 'natural disaster', 'vulnerability', 'adaptation',
    'conservation', 'protection', 'sustainable use', 'ecosystem services',
    'environmental degradation', 'pollution', 'waste management',
    
    # Research & Conservation
    'research station', 'scientific study', 'ecological research', 'marine research',
    'conservation project', 'wildlife protection', 'habitat conservation',
    'species protection', 'breeding program', 'monitoring', 'survey',
    
    # Cultural & Heritage
    'tribal culture', 'traditional knowledge', 'cultural heritage', 'folklore',
    'traditional practices', 'cultural preservation', 'ethnography',
    'anthropological study', 'cultural rights', 'cultural identity',
    
    # Agriculture & Livelihood
    'coconut plantation', 'rubber plantation', 'arecanut', 'paddy cultivation',
    'kitchen garden', 'shifting cultivation', 'traditional agriculture',
    'livelihood', 'subsistence farming', 'fishing community', 'forest produce',
    
    # Government & Administration
    'andaman nicobar administration', 'district collector', 'forest department',
    'tribal welfare', 'panchayat', 'local government', 'revenue village',
    'land records', 'government schemes', 'welfare programs',
    
    # Legal & Policy
    'forest rights act', 'tribal rights', 'environmental clearance',
    'coastal regulation zone', 'crz', 'wildlife protection act',
    'forest conservation act', 'island protection regulation', 'land use policy',
    
    # Challenges & Issues
    'human-elephant conflict', 'human-wildlife conflict', 'encroachment',
    'illegal logging', 'poaching', 'overfishing', 'tourism pressure',
    'waste disposal', 'sewage treatment', 'water scarcity', 'power supply',
    
    # Natural Resources
    'freshwater', 'groundwater', 'natural springs', 'water resources',
    'renewable energy', 'solar energy', 'wind energy', 'natural gas',
    'mineral resources', 'sand mining', 'stone quarrying',
    
    # Weather & Climate
    'tropical climate', 'monsoon', 'rainfall', 'humidity', 'temperature',
    'seasonal variation', 'dry season', 'wet season', 'weather pattern',
    
    # Specific Projects & Initiatives
    'eco-development', 'community-based tourism', 'sustainable tourism',
    'conservation education', 'awareness program', 'capacity building',
    'participatory management', 'stakeholder engagement',
    
    # Infrastructure & Facilities Count/Numbers
    'number of buses', 'bus service', 'public transport', 'bus routes', 'bus frequency',
    'how many buses', 'bus schedule', 'bus timings', 'local transport',
    
    'number of hospitals', 'hospital facilities', 'medical facilities', 'healthcare',
    'how many hospitals', 'primary health center', 'phc', 'dispensary', 'clinic',
    'medical services', 'emergency medical', 'ambulance service',
    
    'number of schools', 'educational institutions', 'primary school', 'high school',
    'how many schools', 'education facilities', 'school enrollment', 'teachers',
    'educational infrastructure', 'literacy rate', 'anganwadi', 'pre-school',
    
    'number of shops', 'retail stores', 'market', 'shopping facilities', 'grocery stores',
    'how many stores', 'local market', 'weekly market', 'cooperative society',
    'fair price shop', 'ration shop', 'commercial establishments',
    
    'number of atms', 'banking facilities', 'bank branches', 'how many atms',
    'financial services', 'post office', 'money transfer', 'digital banking',
    'cooperative bank', 'self help groups', 'microfinance',
    
    'number of villages', 'settlements', 'hamlets', 'revenue villages',
    'how many villages', 'village population', 'household count', 'families',
    'inhabited villages', 'tribal settlements', 'forest villages',
    
    'veterinary hospital', 'vet clinic', 'animal hospital', 'livestock care',
    'veterinary services', 'animal health', 'cattle treatment', 'pet care',
    'veterinary doctor', 'animal welfare', 'vaccination program',
    
    'fire station', 'fire brigade', 'fire service', 'emergency services',
    'fire safety', 'disaster management', 'rescue services', 'fire department',
    
    # Population & Demographics
    'population', 'total population', 'population density', 'demographic data',
    'population census', 'household size', 'age distribution', 'gender ratio',
    'tribal population', 'onge population', 'settler population', 'migrant population',
    'population growth', 'birth rate', 'death rate', 'literacy statistics',
    
    # Area & Land Measurements
    'total area', 'land area', 'geographical area', 'square kilometers', 'sq km',
    'hectares', 'area in hectares', 'island area', 'land measurement',
    'total land', 'usable land', 'cultivable land', 'agricultural area',
    
    'forest area', 'forest cover', 'forest percentage', 'wooded area',
    'reserve forest', 'protected forest', 'forest land', 'tree cover',
    'forest density', 'forest statistics', 'deforestation rate',
    
    'coastal area', 'shoreline length', 'coastline', 'beach area',
    'marine area', 'territorial waters', 'exclusive economic zone',
    
    # Road Infrastructure & Distances
    'road length', 'total road length', 'road network', 'road infrastructure',
    'paved roads', 'unpaved roads', 'motorable roads', 'all weather roads',
    'road condition', 'road connectivity', 'highway', 'state highway',
    'village roads', 'forest roads', 'beach roads',
    
    'distance to port blair', 'distance from port blair', 'distance to hut bay',
    'distance between villages', 'travel distance', 'road distance',
    'aerial distance', 'how far', 'travel time', 'journey time',
    'distance to airport', 'distance to jetty', 'distance to hospital',
    'distance to school', 'nearest town', 'nearest city',
    
    # Utilities & Services Count
    'electricity connections', 'power supply', 'solar installations', 'generators',
    'water supply', 'water connections', 'bore wells', 'hand pumps',
    'water treatment plants', 'sewage treatment', 'waste management',
    
    'telephone connections', 'mobile towers', 'internet connectivity',
    'broadband', 'digital infrastructure', 'communication facilities',
    
    'police station', 'police post', 'security', 'law and order',
    'coast guard', 'border security', 'checkpoints',
    
    # Specific Facility Numbers/Statistics
    'number of beds', 'hospital beds', 'bed capacity', 'icu beds',
    'number of doctors', 'medical staff', 'nurses', 'paramedics',
    'number of teachers', 'student teacher ratio', 'enrollment numbers',
    'number of vehicles', 'registered vehicles', 'two wheelers', 'four wheelers',
    'number of boats', 'fishing boats', 'motorboats', 'traditional boats',
    
    # Economic Data
    'per capita income', 'household income', 'poverty rate', 'employment rate',
    'unemployment', 'income statistics', 'economic indicators', 'gdp',
    'budget allocation', 'government expenditure', 'development funds',
    
    # Fair Price Shops & Public Distribution System
    'fair price shop', 'fps', 'ration shop', 'public distribution system', 'pds',
    'number of fps', 'how many fair price shops', 'ration dealers', 'pds outlets',
    'subsidized food', 'food grains', 'kerosene', 'sugar distribution',
    'monthly quota', 'food security', 'essential commodities',
    
    # Cardholders & Beneficiaries
    'ration cardholders', 'apl cardholders', 'bpl cardholders', 'aay cardholders',
    'above poverty line', 'below poverty line', 'antyodaya anna yojana',
    'number of beneficiaries', 'eligible families', 'cardholder statistics',
    'beneficiary count', 'scheme beneficiaries', 'welfare recipients',
    'food subsidy beneficiaries', 'targeted beneficiaries', 'coverage ratio',
    
    'aadhar cardholders', 'voter id cards', 'identity cards', 'documentation',
    'jan aushadhi beneficiaries', 'health insurance beneficiaries',
    'pension beneficiaries', 'scholarship recipients', 'self help group members',
    
    # Electricity Department & Infrastructure
    'electricity department', 'power department', 'electrical division',
    'power generation capacity', 'installed capacity', 'power consumption',
    'electricity production', 'power demand', 'load shedding', 'power cuts',
    'electrical connections', 'domestic connections', 'commercial connections',
    'industrial connections', 'street lighting', 'power distribution',
    'transformer capacity', 'grid capacity', 'power lines', 'electrical poles',
    'meter readings', 'electricity bills', 'tariff rates', 'power subsidy',
    'renewable energy capacity', 'solar power generation', 'wind power',
    'diesel generators', 'backup power', 'uninterrupted power supply',
    'power outages', 'electrical faults', 'maintenance schedule',
    
    # Water Department & Supply System
    'water department', 'public health engineering', 'phed', 'water supply department',
    'water treatment plant', 'wtp capacity', 'water production capacity',
    'daily water production', 'water storage capacity', 'reservoir capacity',
    'overhead tank capacity', 'underground tank capacity', 'water distribution',
    'pipe network', 'water connections', 'household water connections',
    'commercial water connections', 'institutional connections',
    'water supply hours', 'water quality', 'water testing', 'bacteriological testing',
    'chemical testing', 'water treatment', 'chlorination', 'filtration',
    'water tanker supply', 'emergency water supply', 'water shortage',
    'water conservation', 'rainwater harvesting', 'water recycling',
    'sewage treatment capacity', 'wastewater treatment', 'sewage disposal',
    
    # Fresh Water Resources & Capacity
    'freshwater sources', 'freshwater availability', 'freshwater reserves',
    'groundwater', 'groundwater table', 'water table level', 'aquifer capacity',
    'natural springs', 'stream flow', 'surface water', 'water bodies',
    'pond capacity', 'lake capacity', 'river flow', 'creek flow',
    'well capacity', 'bore well depth', 'bore well yield', 'water extraction',
    'sustainable yield', 'water recharge', 'monsoon recharge', 'infiltration rate',
    'water balance', 'water budget', 'water stress', 'water scarcity',
    'per capita water availability', 'daily water requirement', 'water demand',
    'drinking water', 'potable water', 'safe drinking water', 'water purification',
    'water storage tanks', 'community tanks', 'individual storage',
    
    # Population Sustenance & Carrying Capacity
    'carrying capacity', 'population carrying capacity', 'sustainable population',
    'population pressure', 'overpopulation', 'population limit', 'ecological footprint',
    'resource availability per capita', 'land per capita', 'water per capita',
    'food security', 'food self sufficiency', 'food production capacity',
    'agricultural productivity', 'crop yield', 'food grains production',
    'fish production', 'protein availability', 'nutritional security',
    'calorie availability', 'malnutrition rate', 'undernourishment',
    'food distribution', 'food access', 'food affordability', 'food wastage',
    
    'livelihood sustainability', 'employment capacity', 'job opportunities',
    'income generation', 'economic sustainability', 'resource depletion',
    'environmental degradation', 'ecological balance', 'natural resource management',
    'waste generation', 'waste disposal capacity', 'pollution load',
    'carbon footprint', 'environmental impact per capita',
    
    'healthcare capacity', 'patient load', 'doctor patient ratio',
    'educational capacity', 'student capacity', 'infrastructure load',
    'housing capacity', 'accommodation availability', 'settlement density',
    'transportation capacity', 'traffic load', 'road capacity',
    
    # Government Schemes & Programs
    'mgnrega beneficiaries', 'pmay beneficiaries', 'pradhan mantri awas yojana',
    'swachh bharat mission', 'toilet construction', 'ujjwala yojana',
    'lpg connections', 'ayushman bharat', 'health insurance coverage',
    'pradhan mantri jan dhan yojana', 'bank account holders',
    'digital india', 'skill development', 'startup schemes',
    'kisan credit cards', 'crop insurance', 'pension schemes',
    'widow pension', 'disability pension', 'old age pension',
    
    # Smart Island & NITI Aayog Development Project
    'smart island project', 'smart island initiative', 'digital island', 'smart city',
    'niti aayog proposal', 'niti aayog project', 'niti aayog vision document',
    'sustainable development of little andaman island', 'vision document',
    'little andaman development plan', 'little andaman project', 'mega project',
    'megacity plan', 'megacity project', 'smart island development',
    
    # Greenfield Coastal City Development
    'greenfield coastal city', 'new coastal city', 'planned city', 'modern city',
    'urban development', 'city planning', 'master plan', 'development zones',
    'free trade zone', 'ftz', 'special economic zone', 'sez', 'trade hub',
    'maritime hub', 'startup hub', 'financial district', 'business district',
    'commercial zone', 'industrial zone', 'residential zone', 'tourism zone',
    
    # Three Development Zones
    'zone 1', 'zone 2', 'zone 3', 'development zones', 'zoning plan',
    'financial district', 'medi metropolis', 'aerocity', 'hospital district',
    'leisure zone', 'movie metropolis', 'film city', 'entertainment district',
    'residential areas', 'housing development', 'township development',
    
    # Infrastructure Development Components
    'underwater resorts', 'underwater hotels', 'marine tourism', 'luxury resorts',
    'casinos', 'gaming', 'entertainment complex', 'golf courses', 'sports facilities',
    'convention centers', 'conference facilities', 'exhibition centers',
    'cruise terminals', 'marina development', 'yacht harbors', 'water sports facilities',
    'theme parks', 'amusement parks', 'recreational facilities',
    
    # Connectivity & Transportation
    'airport development', 'runway expansion', 'aviation infrastructure',
    'seaplane services', 'helicopter services', 'inter-island connectivity',
    'ferry services', 'high-speed connectivity', 'broadband infrastructure',
    'digital connectivity', 'submarine cables', 'satellite connectivity',
    'road network expansion', 'highway development', 'bridge construction',
    
    # Comparison with Global Cities
    'singapore model', 'hong kong model', 'compete with singapore', 'compete with hong kong',
    'international trade hub', 'global city', 'world-class infrastructure',
    'international standards', 'global competitiveness', 'trade activity',
    
    # Strategic Location & Geopolitics
    'strategic location', 'indian ocean region', 'ior', 'geopolitical importance',
    'maritime security', 'naval base', 'strategic assets', 'security concerns',
    'china containment', 'look east policy', 'act east policy', 'indo-pacific',
    
    # Project Status & Timeline
    'project status', 'project suspended', 'project cancelled', 'project timeline',
    'implementation plan', 'project phases', 'development stages', 'project funding',
    'investment requirements', 'budget allocation', 'cost estimates', 'financial viability',
    
    # Technology & Innovation
    'smart technology', 'iot implementation', 'artificial intelligence', 'digital governance',
    'e-governance', 'smart utilities', 'smart grid', 'renewable energy integration',
    'waste management technology', 'water management systems', 'traffic management',
    'smart mobility', 'electric vehicles', 'sustainable transport', 'green technology',
    
    # Environmental Concerns & Sustainability
    'environmental impact assessment', 'eia', 'environmental clearance',
    'forest clearance', 'coastal regulation zone clearance', 'crz clearance',
    'biodiversity impact', 'ecological impact', 'carbon footprint', 'green development',
    'sustainable tourism', 'eco-friendly development', 'environmental monitoring',
    'pollution control', 'waste management', 'sewage treatment', 'water treatment',
    
    # Tribal & Social Impact
    'onge tribe impact', 'tribal displacement', 'indigenous rights', 'tribal consultation',
    'rehabilitation', 'resettlement', 'social impact assessment', 'community participation',
    'stakeholder engagement', 'public consultation', 'consent process',
    'cultural preservation', 'traditional livelihood', 'tribal welfare',
    
    # Opposition & Concerns
    'conservationist concerns', 'environmental opposition', 'tribal rights activists',
    'project criticism', 'sustainability concerns', 'ecological concerns',
    'development vs conservation', 'protests', 'legal challenges', 'court cases',
    'ngo opposition', 'civil society concerns', 'expert opinions',
    
    # Economic Aspects
    'economic development', 'gdp contribution', 'employment generation', 'job creation',
    'tourism revenue', 'trade revenue', 'foreign investment', 'fdi',
    'public-private partnership', 'ppp model', 'investment opportunities',
    'economic growth', 'revenue generation', 'tax revenue', 'export earnings',
    
    # Comparative Statistics
    'statistics', 'data', 'figures', 'numbers', 'count', 'total number',
    'how many', 'quantity', 'availability', 'capacity', 'coverage',
    'percentage', 'ratio', 'rate', 'density', 'frequency', 'adequacy',
    'sufficiency', 'shortfall', 'surplus', 'deficit', 'utilization rate'
        ]

        # Cache settings
        self.CACHE_DIR = r"C:\Users\ACER\Documents\NIC_intern\Little Andaman\cache"
        self.CACHE_TTL = 3600  # Cache time-to-live in seconds
        self.ENABLE_CACHE = True
        self.MAX_MEMORY_CACHE = 1000  # Maximum items in memory cache

# Initialize configuration
config = GNIDPConfig()

print("✅ Configuration loaded successfully!")
print(f"📁 PDF Directory: {config.PDF_DIRECTORY}")
print(f"🤖 Ollama Model: {config.OLLAMA_MODEL}")
print(f"🧠 Embedding Model: {config.EMBEDDING_MODEL}")
print(f"📄 Chunk Size: {config.CHUNK_SIZE}")
print(f"🗃️ Vector Store: {config.VECTORSTORE_TYPE}")

# Verify PDF directory exists
if not os.path.exists(config.PDF_DIRECTORY):
    print(f"⚠️  WARNING: PDF directory '{config.PDF_DIRECTORY}' does not exist!")
    print(f"📝 Please create the directory and add your PDF files")
else:
    pdf_files = [f for f in os.listdir(config.PDF_DIRECTORY) if f.endswith('.pdf')]
    print(f"📚 Found {len(pdf_files)} PDF files in directory")


# =====================================
# MODULE 3: EMBEDDING MODEL SETUP
# =====================================
# Initialize the embedding model for document vectorization

def setup_embeddings(config):
    """Initialize embedding model - optimized for speed"""
    logger.info("Loading embedding model...")
    start_time = time.time()
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use GPU if available
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Add batch processing
            }
        )
        
        # Test the embedding model
        test_text = "Great Nicobar Island Development Project"
        test_embedding = embeddings.embed_query(test_text)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Quantized embeddings loaded in {load_time:.2f} seconds")
        logger.info(f"📐 Embedding dimension: {len(test_embedding)}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Error loading embeddings: {str(e)}")
        raise

# Initialize embeddings
print("🧠 Setting up embedding model...")
embeddings = setup_embeddings(config)
print("✅ Embedding model ready!")

# =====================================
# MODULE 4: DOCUMENT LOADING AND PROCESSING
# =====================================
# Load PDFs and create text chunks

from concurrent.futures import ThreadPoolExecutor

def load_and_process_documents(config, force_reload=False):
    """Load PDFs and create text chunks only if needed"""
    
    # Check if vector store exists - using simplified path checks
    faiss_path = os.path.join(config.VECTORSTORE_DIR, f"{config.VECTORSTORE_FILENAME}.faiss")
    pkl_path = os.path.join(config.VECTORSTORE_DIR, f"{config.VECTORSTORE_FILENAME}.pkl")
    
    # Skip document processing if vector store exists and no rebuild requested
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and not force_reload and not config.REBUILD_VECTORSTORE:
        logger.info("📝 Using existing vector store, skipping document processing...")
        return None
    
    logger.info("📖 Loading PDF documents...")
    start_time = time.time()
    
    try:
        # Check if directory exists and has PDFs
        if not os.path.exists(config.PDF_DIRECTORY):
            raise FileNotFoundError(f"PDF directory '{config.PDF_DIRECTORY}' not found!")
        
        pdf_files = [f for f in os.listdir(config.PDF_DIRECTORY) if f.endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{config.PDF_DIRECTORY}'!")
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files")
        
        # Load PDFs in parallel
        def process_pdf(pdf_path):
            loader = PyPDFLoader(pdf_path)
            return loader.load()
            
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_pdf, os.path.join(config.PDF_DIRECTORY, pdf)) 
                      for pdf in pdf_files]
            documents = [doc for future in futures for doc in future.result()]
            
        logger.info(f"📄 Loaded {len(documents)} document pages")
        
        # Split documents into chunks with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", ": ", " ", ""],
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Display some statistics
        total_chars = sum(len(doc.page_content) for doc in texts)
        avg_chunk_size = total_chars // len(texts) if texts else 0
        
        load_time = time.time() - start_time
        logger.info(f"✂️  Created {len(texts)} text chunks")
        logger.info(f"📊 Average chunk size: {avg_chunk_size} characters")
        logger.info(f"⏱️  Document processing completed in {load_time:.2f} seconds")
        
        return texts
        
    except Exception as e:
        logger.error(f"❌ Error processing documents: {str(e)}")
        raise

# Process documents only if needed
print("📚 Checking document processing requirements...")
documents = load_and_process_documents(config)
if documents:
    print(f"✅ Successfully processed {len(documents)} text chunks!")
    # Display first chunk as sample
    print("\n📖 Sample chunk:")
    print("-" * 50)
    print(documents[0].page_content[:300] + "...")
    print("-" * 50)
else:
    print("✅ Using existing vector store, document processing skipped!")


# =====================================
# MODULE 5: VECTOR STORE CREATION
# =====================================
# Create and setup the vector database

global vectorstore

def create_or_load_vectorstore(texts, embeddings, config):
    """Create or load vector store with persistence"""
    global vectorstore
    # Create directory if it doesn't exist
    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    
    logger.info(f"🗃️ {'Creating' if config.REBUILD_VECTORSTORE else 'Loading'} vector store...")
    start_time = time.time()
    
    faiss_path = os.path.join(config.VECTORSTORE_DIR, f"{config.VECTORSTORE_FILENAME}.faiss")
    pkl_path = os.path.join(config.VECTORSTORE_DIR, f"{config.VECTORSTORE_FILENAME}.pkl")
    
    # Try loading existing vector store first if not rebuilding
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and not config.REBUILD_VECTORSTORE:
        try:
            logger.info("🔄 Loading existing vector store...")
            vectorstore = FAISS.load_local(
                folder_path=config.VECTORSTORE_DIR,
                embeddings=embeddings,
                index_name=config.VECTORSTORE_FILENAME
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Vector store loaded successfully in {load_time:.2f} seconds")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {str(e)}")
            logger.info("🔄 Falling back to creating new vector store")
            config.REBUILD_VECTORSTORE = True
    
    # Create new vector store if loading failed, rebuild requested, or files don't exist
    if texts and (config.REBUILD_VECTORSTORE or not (os.path.exists(faiss_path) and os.path.exists(pkl_path))):
        logger.info("🏗️ Creating new vector store...")
        vectorstore = FAISS.from_documents(
            texts, 
            embeddings,
            distance_strategy="COSINE"
        )
            
        # Save the vector store
        try:
            vectorstore.save_local(config.VECTORSTORE_DIR, config.VECTORSTORE_FILENAME)
            logger.info(f"💾 Vector store saved successfully")
        except Exception as e:
            logger.error(f"❌ Error saving vector store: {str(e)}")
            raise
        
        creation_time = time.time() - start_time
        logger.info(f"✅ Vector store creation completed in {creation_time:.2f} seconds")
        
        return vectorstore
    
    logger.error("❌ Vector store not found and no documents provided to create new one")
    raise FileNotFoundError("Vector store files not found and no documents available to create new one")

# Create or load vector store
print("🗃️ Creating/loading vector database...")

try:
    vectorstore = create_or_load_vectorstore(documents, embeddings, config)
    print("✅ Vector store ready for queries!")

    # Test the vector store
    print("\n🔍 Testing vector store with sample query...")
    test_query = "environmental impact of GNIDP"
    test_results = vectorstore.similarity_search(test_query, k=2)

    print(f"📊 Found {len(test_results)} relevant documents")
    if test_results:
        print("\n📄 Most relevant chunk:")
        print("-" * 50)
        print(test_results[0].page_content[:200] + "...")
        print("-" * 50)
except Exception as e:
    print(f"❌ Error with vector store: {str(e)}")
    print("🔧 Try setting config.REBUILD_VECTORSTORE = True to rebuild the vector store")

# =====================================
# MODULE 6: LLM SETUP AND CONFIGURATION
# =====================================
# Initialize Ollama LLM and create custom prompt template

def clean_response(text: str) -> str:
    """Clean up model response by removing XML-like tags and extra whitespace"""
    import re
    
    # Remove XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up extra whitespace
    text = text.strip()
    
    return text

def setup_llm(config):
    """Initialize Ollama LLM"""
    logger.info("🤖 Setting up Ollama LLM...")
    start_time = time.time()

    try:
        llm = Ollama(
            model=config.OLLAMA_MODEL,
            temperature=config.TEMPERATURE,
            num_ctx=config.NUM_CTX,
            num_predict=512,  # Limit response length
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            repeat_penalty=1.1,  # Prevent repetitive responses
            format="json"  # Force structured output
        )

        # Test the LLM connection
        test_response = llm.invoke("Hello, are you working?")

        setup_time = time.time() - start_time
        logger.info(f"✅ LLM setup completed in {setup_time:.2f} seconds")
        logger.info(f"🎯 Model: {config.OLLAMA_MODEL} (GPU-accelerated)")
        logger.info(f"🌡️  Temperature: {config.TEMPERATURE}")

        return llm

    except Exception as e:
        logger.error(f"❌ Error setting up LLM: {str(e)}")
        logger.error("🔧 Make sure Ollama is running: ollama serve")
        raise


def create_prompt_template():
    """Create custom prompt template for GNIDP-focused responses"""
    
    template = """You are an expert assistant specialized in Little Andaman Island, its development projects, infrastructure, demographics, and all matters related to the Andaman & Nicobar Islands administration.

        Gather information only from the provided context and documents to give a proper structured answer to the queries. 
        Context from Knowledge Base: {context}

        User Question: {question}
        Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    logger.info("📝 Custom prompt template created")
    return prompt

# Setup LLM
print("🤖 Initializing Ollama LLM...")
llm = setup_llm(config)

# Create prompt template
print("📝 Creating custom prompt template...")
prompt_template = create_prompt_template()

print("✅ LLM and prompt template ready!")
