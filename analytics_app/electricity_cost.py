import time
import psutil
import GPUtil
import logging
from django.conf import settings
from django.utils import timezone
from datetime import timedelta, datetime
import numpy as np
import threading
import json
import os
from pathlib import Path
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)

class ElectricityCostTracker:
    """
    Tracks and forecasts electricity costs for running local models
    with enhanced support for GPU monitoring and vision model tracking
    """
    
    def __init__(self, user=None):
        """Initialize with user-specific settings if available"""
        # Initialize tracking data
        self.usage_history = []
        self.max_history_items = 100  # Keep the last 100 inferences for averaging
        
        # GPU timeline tracking
        self.gpu_timeline = []
        self.gpu_tracking_enabled = False
        self.gpu_sampling_interval = 2.0  # seconds
        self.gpu_tracker_thread = None
        self.gpu_tracking_lock = threading.Lock()
        self.timeline_max_samples = 1000  # Maximum samples to keep
        
        # Vision model specific tracking
        self.vision_usage_history = []  # Separate tracking for vision models
        
        # Try to load user-specific settings if user is provided
        if user and user.is_authenticated:
            try:
                from .models import ElectricitySettings
                user_settings = ElectricitySettings.objects.get(user=user)
                self.kwh_rate = float(user_settings.kwh_rate)
                self.gpu_idle_watts = user_settings.gpu_idle_watts
                self.gpu_load_watts = user_settings.gpu_load_watts
                self.cpu_idle_watts = user_settings.cpu_idle_watts
                self.cpu_load_watts = user_settings.cpu_load_watts
                return
            except Exception as e:
                logger.warning(f"Could not load user electricity settings: {e}")
        
        # Default values if user settings not available
        self.kwh_rate = getattr(settings, 'ELECTRICITY_KWH_RATE', 0.24)  # Default 0.24 CHF per kWh
        self.gpu_idle_watts = getattr(settings, 'GPU_IDLE_WATTS', 15)    # Watts when idle
        self.gpu_load_watts = getattr(settings, 'GPU_LOAD_WATTS', 200)   # Watts at full load
        self.cpu_idle_watts = getattr(settings, 'CPU_IDLE_WATTS', 10)    # Watts per core when idle
        self.cpu_load_watts = getattr(settings, 'CPU_LOAD_WATTS', 35)    # Watts per core at full load
        
        # GPU model-specific power profiles
        self.gpu_power_profiles = getattr(settings, 'GPU_POWER_PROFILES', {
            'RTX 3090': {'idle_watts': 30, 'load_watts': 350, 'vram': 24},
            'RTX 3080': {'idle_watts': 25, 'load_watts': 320, 'vram': 10},
            'RTX 3070': {'idle_watts': 20, 'load_watts': 220, 'vram': 8},
            'RTX 3060': {'idle_watts': 15, 'load_watts': 170, 'vram': 12},
            'RTX 2080 Ti': {'idle_watts': 20, 'load_watts': 280, 'vram': 11},
            'RTX 2080': {'idle_watts': 18, 'load_watts': 225, 'vram': 8},
            'GTX 1080 Ti': {'idle_watts': 15, 'load_watts': 250, 'vram': 11},
            'GTX 1080': {'idle_watts': 12, 'load_watts': 180, 'vram': 8},
            'T4': {'idle_watts': 8, 'load_watts': 70, 'vram': 16},
            'A100': {'idle_watts': 40, 'load_watts': 400, 'vram': 40}
        })
        
        # Load cloud vision API costs
        self.cloud_vision_costs = getattr(settings, 'CLOUD_VISION_COSTS', {
            'openai_gpt4v': 0.00765,  # per image
            'gemini_vision_pro': 0.0035,  # per image
            'claude3_vision': 0.00675,  # per image
            'qwen_vl': 0.002  # per image
        })
    
    def start_tracking(self, model_id, num_tokens=1000, is_vision_model=False, num_images=0):
        """Start tracking resource usage for a model inference"""
        tracking_data = {
            'model_id': model_id,
            'num_tokens': num_tokens,
            'is_vision_model': is_vision_model,
            'num_images': num_images,
            'start_time': time.time(),
            'start_cpu': psutil.cpu_percent(interval=0.1),
            'gpu_available': len(GPUtil.getGPUs()) > 0,
            'start_timestamp': timezone.now()
        }
        
        # Get GPU usage if available
        if tracking_data['gpu_available']:
            gpus = GPUtil.getGPUs()
            tracking_data['start_gpu'] = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            tracking_data['gpu_memory'] = sum(gpu.memoryTotal for gpu in gpus)
            tracking_data['gpu_memory_used'] = sum(gpu.memoryUsed for gpu in gpus)
            tracking_data['gpu_models'] = [gpu.name for gpu in gpus]
            
            # Get CUDA memory info if torch is available
            if torch.cuda.is_available():
                tracking_data['cuda_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                tracking_data['cuda_memory_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        return tracking_data
    
    def stop_tracking(self, tracking_data):
        """Stop tracking and calculate resource usage & costs"""
        end_time = time.time()
        tracking_data['end_timestamp'] = timezone.now()
        tracking_data['duration'] = end_time - tracking_data['start_time']
        
        # CPU usage
        end_cpu = psutil.cpu_percent(interval=0.1)
        avg_cpu = (tracking_data['start_cpu'] + end_cpu) / 2
        tracking_data['avg_cpu_percent'] = avg_cpu
        
        # Calculate CPU power consumption (in watts)
        cpu_count = psutil.cpu_count()
        cpu_power = (self.cpu_idle_watts * cpu_count) + (self.cpu_load_watts - self.cpu_idle_watts) * cpu_count * (avg_cpu / 100)
        tracking_data['cpu_power_watts'] = cpu_power
        
        # GPU usage if available
        if tracking_data.get('gpu_available', False):
            gpus = GPUtil.getGPUs()
            end_gpu = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            avg_gpu = (tracking_data['start_gpu'] + end_gpu) / 2
            tracking_data['avg_gpu_percent'] = avg_gpu
            
            # Calculate GPU power consumption (in watts) - use specific profiles if available
            gpu_power = 0
            for i, gpu_name in enumerate(tracking_data.get('gpu_models', [])):
                # Find the closest matching GPU profile
                matched_profile = None
                for profile_name, profile in self.gpu_power_profiles.items():
                    if profile_name in gpu_name:
                        matched_profile = profile
                        break
                
                if matched_profile:
                    # Use matched profile
                    idle_watts = matched_profile['idle_watts']
                    load_watts = matched_profile['load_watts']
                else:
                    # Use default values
                    idle_watts = self.gpu_idle_watts
                    load_watts = self.gpu_load_watts
                
                # Get specific GPU load if available
                if i < len(gpus):
                    gpu_util = gpus[i].load * 100
                else:
                    gpu_util = avg_gpu
                
                # Calculate GPU power
                gpu_power += idle_watts + (load_watts - idle_watts) * (gpu_util / 100)
            
            tracking_data['gpu_power_watts'] = gpu_power
            
            # Also check GPU memory at end
            tracking_data['end_gpu_memory_used'] = sum(gpu.memoryUsed for gpu in gpus)
            tracking_data['gpu_memory_diff'] = tracking_data['end_gpu_memory_used'] - tracking_data.get('gpu_memory_used', 0)
            
            # Get CUDA memory info at end if torch is available
            if torch.cuda.is_available():
                tracking_data['end_cuda_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                tracking_data['end_cuda_memory_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                tracking_data['cuda_memory_allocated_diff'] = tracking_data['end_cuda_memory_allocated'] - tracking_data.get('cuda_memory_allocated', 0)
            
            # Total power is CPU + GPU
            total_power = cpu_power + gpu_power
        else:
            # Just CPU power if no GPU
            total_power = cpu_power
        
        tracking_data['total_power_watts'] = total_power
        
        # Calculate electricity cost
        # Convert watts to kilowatts and hours to get kWh
        kwh_used = (total_power / 1000) * (tracking_data['duration'] / 3600)
        tracking_data['kwh_used'] = kwh_used
        tracking_data['electricity_cost'] = kwh_used * self.kwh_rate
        
        # Calculate tokens per second
        if tracking_data['duration'] > 0:
            tracking_data['tokens_per_second'] = tracking_data['num_tokens'] / tracking_data['duration']
        else:
            tracking_data['tokens_per_second'] = 0
        
        # Calculate cost per 1000 tokens
        if tracking_data['num_tokens'] > 0:
            tracking_data['cost_per_1k_tokens'] = (tracking_data['electricity_cost'] / tracking_data['num_tokens']) * 1000
        else:
            tracking_data['cost_per_1k_tokens'] = 0
        
        # If it's a vision model, also calculate cost per image
        if tracking_data.get('is_vision_model', False) and tracking_data.get('num_images', 0) > 0:
            tracking_data['cost_per_image'] = tracking_data['electricity_cost'] / tracking_data['num_images']
            # Add to vision history
            self.vision_usage_history.append(tracking_data)
            if len(self.vision_usage_history) > self.max_history_items:
                self.vision_usage_history = self.vision_usage_history[-self.max_history_items:]
        
        # Add to history and trim if needed
        self.usage_history.append(tracking_data)
        if len(self.usage_history) > self.max_history_items:
            self.usage_history = self.usage_history[-self.max_history_items:]
        
        return tracking_data
    
    def start_gpu_monitoring(self):
        """
        Start continuous monitoring of GPU usage and power consumption.
        Creates a background thread that samples GPU metrics at regular intervals.
        """
        if self.gpu_tracking_enabled:
            return  # Already tracking
            
        with self.gpu_tracking_lock:
            self.gpu_timeline = []
            self.gpu_tracking_enabled = True
            
            # Define the tracking function
            def track_gpu():
                start_time = time.time()
                
                while self.gpu_tracking_enabled:
                    try:
                        # Get current timestamp
                        current_time = time.time() - start_time
                        
                        # System memory
                        memory = psutil.virtual_memory()
                        
                        # GPU metrics
                        gpus = GPUtil.getGPUs()
                        gpu_data = []
                        total_gpu_power = 0
                        
                        for i, gpu in enumerate(gpus):
                            # Find power profile
                            gpu_profile = None
                            for profile_name, profile in self.gpu_power_profiles.items():
                                if profile_name in gpu.name:
                                    gpu_profile = profile
                                    break
                            
                            if not gpu_profile:
                                # Use default values
                                idle_watts = self.gpu_idle_watts
                                load_watts = self.gpu_load_watts
                            else:
                                idle_watts = gpu_profile['idle_watts']
                                load_watts = gpu_profile['load_watts']
                            
                            # Calculate estimated power
                            gpu_util = gpu.load * 100
                            gpu_power = idle_watts + (load_watts - idle_watts) * (gpu_util / 100)
                            total_gpu_power += gpu_power
                            
                            gpu_info = {
                                'index': i,
                                'name': gpu.name,
                                'utilization': gpu_util,
                                'memory_used': gpu.memoryUsed,
                                'memory_total': gpu.memoryTotal,
                                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                                'temperature': gpu.temperature,
                                'estimated_power_watts': gpu_power
                            }
                            
                            # Add CUDA specific info if available
                            if torch.cuda.is_available() and i < torch.cuda.device_count():
                                try:
                                    gpu_info['cuda_memory_allocated'] = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                                    gpu_info['cuda_memory_reserved'] = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                                except:
                                    pass
                                    
                            gpu_data.append(gpu_info)
                        
                        # Calculate electricity cost rate (per hour)
                        hourly_cost = (total_gpu_power / 1000) * self.kwh_rate
                        
                        # Collect all metrics
                        timeline_point = {
                            'timestamp': current_time,
                            'datetime': datetime.now().isoformat(),
                            'system_memory': {
                                'total': memory.total / (1024 ** 2),  # MB
                                'available': memory.available / (1024 ** 2),  # MB
                                'percent': memory.percent
                            },
                            'gpus': gpu_data,
                            'total_gpu_power_watts': total_gpu_power,
                            'hourly_cost': hourly_cost,
                            'daily_cost_estimate': hourly_cost * 24
                        }
                        
                        # Add to timeline
                        with self.gpu_tracking_lock:
                            self.gpu_timeline.append(timeline_point)
                            # Trim if needed
                            if len(self.gpu_timeline) > self.timeline_max_samples:
                                self.gpu_timeline = self.gpu_timeline[-self.timeline_max_samples:]
                    
                    except Exception as e:
                        logger.error(f"Error in GPU tracking: {e}")
                    
                    # Sleep for the sampling interval
                    time.sleep(self.gpu_sampling_interval)
            
            # Start tracking thread
            self.gpu_tracker_thread = threading.Thread(
                target=track_gpu, 
                daemon=True,  # Daemon thread will exit when main thread exits
                name="GPUPowerTracker"
            )
            self.gpu_tracker_thread.start()
            logger.info("Started GPU power monitoring")
            
            return True
    
    def stop_gpu_monitoring(self):
        """
        Stop GPU monitoring and return the collected timeline.
        """
        if not self.gpu_tracking_enabled:
            return self.gpu_timeline
            
        with self.gpu_tracking_lock:
            self.gpu_tracking_enabled = False
            
            # Wait for thread to finish if it exists
            if self.gpu_tracker_thread and self.gpu_tracker_thread.is_alive():
                self.gpu_tracker_thread.join(timeout=2.0)
                
            # Clone the timeline to return
            timeline = list(self.gpu_timeline)
            logger.info(f"Stopped GPU power monitoring. Collected {len(timeline)} samples.")
            
            return timeline
    
    def save_gpu_timeline(self, file_path=None):
        """
        Save the GPU timeline to a JSON file for later analysis.
        
        Args:
            file_path: Optional file path, defaults to a timestamped file in logs/gpu_power/
        
        Returns:
            str: Path to the saved file
        """
        with self.gpu_tracking_lock:
            if not self.gpu_timeline:
                return None
            
            # Create default path if not provided
            if file_path is None:
                logs_dir = Path(settings.BASE_DIR) / "logs" / "gpu_power"
                os.makedirs(logs_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = logs_dir / f"gpu_timeline_{timestamp}.json"
            
            # Convert path to string if it's a Path object
            if isinstance(file_path, Path):
                file_path = str(file_path)
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.gpu_timeline, f)
                logger.info(f"Saved GPU timeline with {len(self.gpu_timeline)} samples to {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Error saving GPU timeline: {e}")
                return None
    
    def get_model_stats(self, model_id):
        """Get usage statistics for a specific model"""
        model_history = [item for item in self.usage_history if item['model_id'] == model_id]
        
        if not model_history:
            return None
        
        stats = {
            'model_id': model_id,
            'total_runs': len(model_history),
            'avg_duration': sum(item['duration'] for item in model_history) / len(model_history),
            'avg_tokens_per_second': sum(item.get('tokens_per_second', 0) for item in model_history) / len(model_history),
            'avg_electricity_cost': sum(item['electricity_cost'] for item in model_history) / len(model_history),
            'avg_cost_per_1k_tokens': sum(item.get('cost_per_1k_tokens', 0) for item in model_history) / len(model_history),
            'total_electricity_cost': sum(item['electricity_cost'] for item in model_history),
            'total_kwh_used': sum(item['kwh_used'] for item in model_history),
        }
        
        # Add GPU stats if available
        gpu_history = [item for item in model_history if item.get('gpu_available', False)]
        if gpu_history:
            stats['gpu_stats'] = {
                'avg_gpu_percent': sum(item.get('avg_gpu_percent', 0) for item in gpu_history) / len(gpu_history),
                'avg_gpu_power_watts': sum(item.get('gpu_power_watts', 0) for item in gpu_history) / len(gpu_history),
                'avg_gpu_memory_used': sum(item.get('gpu_memory_used', 0) for item in gpu_history) / len(gpu_history),
            }
        
        return stats
    
    def get_vision_model_stats(self, model_id=None):
        """
        Get usage statistics for vision models
        
        Args:
            model_id: Optional model ID to filter, if None returns stats for all vision models
        
        Returns:
            dict: Statistics for vision model(s)
        """
        if model_id:
            vision_history = [item for item in self.vision_usage_history 
                             if item['model_id'] == model_id]
        else:
            vision_history = self.vision_usage_history
        
        if not vision_history:
            return None
        
        # Group by model ID
        models = {}
        for item in vision_history:
            model = item['model_id']
            if model not in models:
                models[model] = []
            models[model].append(item)
        
        # Calculate stats for each model
        model_stats = {}
        for model, history in models.items():
            model_stats[model] = {
                'total_runs': len(history),
                'avg_duration': sum(item['duration'] for item in history) / len(history),
                'avg_electricity_cost': sum(item['electricity_cost'] for item in history) / len(history),
                'total_electricity_cost': sum(item['electricity_cost'] for item in history),
                'total_kwh_used': sum(item['kwh_used'] for item in history),
                'avg_cost_per_image': sum(item.get('cost_per_image', 0) for item in history) / len(history),
                'total_images_processed': sum(item.get('num_images', 0) for item in history),
            }
            
            # Add GPU stats if available
            gpu_history = [item for item in history if item.get('gpu_available', False)]
            if gpu_history:
                model_stats[model]['gpu_stats'] = {
                    'avg_gpu_percent': sum(item.get('avg_gpu_percent', 0) for item in gpu_history) / len(gpu_history),
                    'avg_gpu_power_watts': sum(item.get('gpu_power_watts', 0) for item in gpu_history) / len(gpu_history),
                    'avg_gpu_memory_used': sum(item.get('gpu_memory_used', 0) for item in gpu_history) / len(gpu_history),
                }
        
        # Overall stats
        overall = {
            'models': model_stats,
            'total_models': len(models),
            'total_runs': len(vision_history),
            'avg_duration': sum(item['duration'] for item in vision_history) / len(vision_history),
            'avg_electricity_cost': sum(item['electricity_cost'] for item in vision_history) / len(vision_history),
            'total_electricity_cost': sum(item['electricity_cost'] for item in vision_history),
            'total_kwh_used': sum(item['kwh_used'] for item in vision_history),
            'total_images_processed': sum(item.get('num_images', 0) for item in vision_history),
        }
        
        if model_id and model_id in model_stats:
            return model_stats[model_id]
        
        return overall
    
    def forecast_cost(self, model_id, token_count, daily_queries=100):
        """Forecast electricity costs for a given usage pattern"""
        model_stats = self.get_model_stats(model_id)
        
        if not model_stats:
            return None
        
        # Calculate costs at different time scales
        cost_per_1k = model_stats['avg_cost_per_1k_tokens']
        tokens_per_query = token_count
        
        forecast = {
            'cost_per_query': (tokens_per_query / 1000) * cost_per_1k,
            'daily_cost': (tokens_per_query / 1000) * cost_per_1k * daily_queries,
            'monthly_cost': (tokens_per_query / 1000) * cost_per_1k * daily_queries * 30,
            'yearly_cost': (tokens_per_query / 1000) * cost_per_1k * daily_queries * 365,
        }
        
        # Compare with typical cloud API costs
        # These are example rates, real ones would come from settings or APIs
        cloud_costs = {
            'openai_gpt35': 0.002,  # per 1K tokens
            'openai_gpt4': 0.06,    # per 1K tokens
            'claude3_haiku': 0.0025, # per 1K tokens
            'claude3_opus': 0.03,   # per 1K tokens
        }
        
        cloud_comparison = {}
        for provider, rate in cloud_costs.items():
            provider_cost = (tokens_per_query / 1000) * rate
            cloud_comparison[provider] = {
                'cost_per_query': provider_cost,
                'daily_cost': provider_cost * daily_queries,
                'monthly_cost': provider_cost * daily_queries * 30,
                'yearly_cost': provider_cost * daily_queries * 365,
                'comparison': {
                    'per_query': provider_cost - forecast['cost_per_query'],
                    'daily': (provider_cost * daily_queries) - forecast['daily_cost'],
                    'monthly': (provider_cost * daily_queries * 30) - forecast['monthly_cost'],
                    'yearly': (provider_cost * daily_queries * 365) - forecast['yearly_cost'],
                }
            }
        
        forecast['cloud_comparison'] = cloud_comparison
        
        return forecast

    def forecast_vision_cost(self, model_id, daily_images=50):
        """
        Forecast electricity costs for a vision model based on image processing volume
        
        Args:
            model_id: Vision model ID
            daily_images: Number of images processed daily
        
        Returns:
            dict: Cost forecast and cloud comparison
        """
        model_stats = self.get_vision_model_stats(model_id)
        
        if not model_stats:
            return None
        
        # Calculate costs at different time scales
        cost_per_image = model_stats.get('avg_cost_per_image', 0)
        
        forecast = {
            'cost_per_image': cost_per_image,
            'daily_cost': cost_per_image * daily_images,
            'monthly_cost': cost_per_image * daily_images * 30,
            'yearly_cost': cost_per_image * daily_images * 365,
        }
        
        # Compare with cloud vision API costs
        cloud_comparison = {}
        for provider, rate in self.cloud_vision_costs.items():
            provider_cost = rate  # Cost per image
            cloud_comparison[provider] = {
                'cost_per_image': provider_cost,
                'daily_cost': provider_cost * daily_images,
                'monthly_cost': provider_cost * daily_images * 30,
                'yearly_cost': provider_cost * daily_images * 365,
                'comparison': {
                    'per_image': provider_cost - forecast['cost_per_image'],
                    'daily': (provider_cost * daily_images) - forecast['daily_cost'],
                    'monthly': (provider_cost * daily_images * 30) - forecast['monthly_cost'],
                    'yearly': (provider_cost * daily_images * 365) - forecast['yearly_cost'],
                }
            }
        
        forecast['cloud_comparison'] = cloud_comparison
        
        # Calculate potential savings with local model
        total_cloud_costs = {}
        for provider, data in cloud_comparison.items():
            yearly_cost = data['yearly_cost']
            yearly_diff = data['comparison']['yearly']
            if yearly_diff > 0:  # If cloud is more expensive
                total_cloud_costs[provider] = {
                    'yearly_cost': yearly_cost,
                    'yearly_savings': yearly_diff,
                    'percent_savings': (yearly_diff / yearly_cost) * 100 if yearly_cost > 0 else 0
                }
        
        forecast['potential_savings'] = total_cloud_costs
        
        return forecast
        
    def get_gpu_power_analysis(self):
        """
        Analyze GPU power usage from the timeline data
        
        Returns:
            dict: Analysis results
        """
        with self.gpu_tracking_lock:
            if not self.gpu_timeline:
                return None
            
            timeline = list(self.gpu_timeline)
            
        # Calculate overall stats
        total_power_readings = []
        gpu_utils = defaultdict(list)
        gpu_temps = defaultdict(list)
        gpu_memory = defaultdict(list)
        hourly_costs = []
        
        for point in timeline:
            # Overall power
            total_power_readings.append(point.get('total_gpu_power_watts', 0))
            hourly_costs.append(point.get('hourly_cost', 0))
            
            # Per-GPU stats
            for gpu in point.get('gpus', []):
                gpu_index = gpu.get('index', 0)
                gpu_utils[gpu_index].append(gpu.get('utilization', 0))
                gpu_temps[gpu_index].append(gpu.get('temperature', 0))
                gpu_memory[gpu_index].append(gpu.get('memory_percent', 0))
        
        # Calculate summary statistics
        avg_power = sum(total_power_readings) / max(len(total_power_readings), 1)
        max_power = max(total_power_readings) if total_power_readings else 0
        min_power = min(total_power_readings) if total_power_readings else 0
        
        avg_hourly_cost = sum(hourly_costs) / max(len(hourly_costs), 1)
        daily_cost_estimate = avg_hourly_cost * 24
        monthly_cost_estimate = daily_cost_estimate * 30
        
        # Per-GPU statistics
        gpu_stats = {}
        for gpu_index in gpu_utils.keys():
            utils = gpu_utils[gpu_index]
            temps = gpu_temps[gpu_index]
            mems = gpu_memory[gpu_index]
            
            gpu_stats[f"gpu_{gpu_index}"] = {
                'avg_utilization': sum(utils) / max(len(utils), 1),
                'max_utilization': max(utils) if utils else 0,
                'avg_temperature': sum(temps) / max(len(temps), 1),
                'max_temperature': max(temps) if temps else 0,
                'avg_memory_percent': sum(mems) / max(len(mems), 1),
                'max_memory_percent': max(mems) if mems else 0,
            }
        
        # Duration of monitoring
        duration_seconds = 0
        if len(timeline) >= 2:
            first_timestamp = timeline[0].get('timestamp', 0)
            last_timestamp = timeline[-1].get('timestamp', 0)
            duration_seconds = last_timestamp - first_timestamp
        
        # Create result
        result = {
            'monitoring_duration_seconds': duration_seconds,
            'monitoring_duration_hours': duration_seconds / 3600,
            'samples_collected': len(timeline),
            'power_stats': {
                'avg_watts': avg_power,
                'max_watts': max_power,
                'min_watts': min_power,
            },
            'cost_estimates': {
                'avg_hourly_cost': avg_hourly_cost,
                'daily_cost_estimate': daily_cost_estimate,
                'monthly_cost_estimate': monthly_cost_estimate,
                'yearly_cost_estimate': monthly_cost_estimate * 12,
            },
            'gpu_stats': gpu_stats
        }
        
        # Add power distribution
        if total_power_readings:
            # Calculate percentiles
            percentiles = {
                'p25': np.percentile(total_power_readings, 25),
                'p50': np.percentile(total_power_readings, 50),
                'p75': np.percentile(total_power_readings, 75),
                'p90': np.percentile(total_power_readings, 90),
                'p95': np.percentile(total_power_readings, 95),
                'p99': np.percentile(total_power_readings, 99),
            }
            result['power_percentiles'] = percentiles
            
            # Check for power spikes
            power_threshold = percentiles['p95'] * 1.5
            spikes = [reading for reading in total_power_readings if reading > power_threshold]
            result['power_spikes'] = {
                'count': len(spikes),
                'avg_spike_watts': sum(spikes) / max(len(spikes), 1) if spikes else 0,
                'max_spike_watts': max(spikes) if spikes else 0,
            }
        
        return result 