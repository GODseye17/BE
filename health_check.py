#!/usr/bin/env python3
"""
Simple Health Check Script for Vivum RAG Backend
Quick verification of system status
"""
import asyncio
import time
import requests
import sys
from typing import Dict, Any

def check_server_status() -> Dict[str, Any]:
    """Check if the server is running"""
    print("🏥 Checking Server Status...")
    start_time = time.time()
    
    try:
        from config.settings import PORT
        
        # Check root endpoint
        response = requests.get(f"http://localhost:{PORT}/", timeout=5)
        
        if response.status_code == 200:
            duration = time.time() - start_time
            print(f"   ✅ Server is running ({duration:.2f}s)")
            return {
                "status": "success",
                "message": "Server is running",
                "duration": duration
            }
        else:
            duration = time.time() - start_time
            print(f"   ❌ Server returned status {response.status_code}")
            return {
                "status": "error",
                "message": f"Server returned status {response.status_code}",
                "duration": duration
            }
            
    except requests.exceptions.ConnectionError:
        duration = time.time() - start_time
        print(f"   ❌ Server is not running")
        return {
            "status": "error",
            "message": "Server is not running",
            "duration": duration
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Error checking server: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duration": duration
        }

def check_supabase_status() -> Dict[str, Any]:
    """Check Supabase connection"""
    print("🏥 Checking Supabase Status...")
    start_time = time.time()
    
    try:
        from config.settings import PORT
        
        response = requests.get(f"http://localhost:{PORT}/supabase-status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            duration = time.time() - start_time
            
            if data.get("status") == "connected":
                print(f"   ✅ Supabase connected ({duration:.2f}s)")
                return {
                    "status": "success",
                    "message": "Supabase connected",
                    "duration": duration
                }
            else:
                print(f"   ❌ Supabase not connected: {data.get('message')}")
                return {
                    "status": "error",
                    "message": data.get('message', 'Unknown error'),
                    "duration": duration
                }
        else:
            duration = time.time() - start_time
            print(f"   ❌ Supabase check failed: {response.status_code}")
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "duration": duration
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Error checking Supabase: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duration": duration
        }

def check_model_status() -> Dict[str, Any]:
    """Check model status"""
    print("🏥 Checking Model Status...")
    start_time = time.time()
    
    try:
        from config.settings import PORT
        
        response = requests.get(f"http://localhost:{PORT}/model-status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            duration = time.time() - start_time
            
            embedding_status = data.get("embedding_model", "unknown")
            llm_status = data.get("llm", "unknown")
            
            print(f"   ✅ Embedding model: {embedding_status} ({duration:.2f}s)")
            print(f"   ✅ LLM model: {llm_status}")
            
            if embedding_status == "loaded" and llm_status == "loaded":
                return {
                    "status": "success",
                    "message": "All models loaded",
                    "duration": duration
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Embedding: {embedding_status}, LLM: {llm_status}",
                    "duration": duration
                }
        else:
            duration = time.time() - start_time
            print(f"   ❌ Model check failed: {response.status_code}")
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "duration": duration
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Error checking models: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duration": duration
        }

def check_system_health() -> Dict[str, Any]:
    """Check system health"""
    print("🏥 Checking System Health...")
    start_time = time.time()
    
    try:
        from config.settings import PORT
        
        response = requests.get(f"http://localhost:{PORT}/system-health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            duration = time.time() - start_time
            
            health = data.get('system_health', {})
            cpu_usage = health.get('cpu_usage', 0)
            memory_usage = health.get('memory_usage', 0)
            disk_usage = health.get('disk_usage', 0)
            
            print(f"   ✅ CPU: {cpu_usage:.1f}% ({duration:.2f}s)")
            print(f"   ✅ Memory: {memory_usage:.1f}%")
            print(f"   ✅ Disk: {disk_usage:.1f}%")
            
            # Check if usage is reasonable
            if cpu_usage < 90 and memory_usage < 90 and disk_usage < 90:
                return {
                    "status": "success",
                    "message": "System health good",
                    "duration": duration,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                }
            else:
                return {
                    "status": "warning",
                    "message": f"High usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%",
                    "duration": duration,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                }
        else:
            duration = time.time() - start_time
            print(f"   ❌ Health check failed: {response.status_code}")
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}",
                "duration": duration
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Error checking system health: {e}")
        return {
            "status": "error",
            "message": str(e),
            "duration": duration
        }

def main():
    """Main function"""
    print("🏥 Vivum RAG Backend Health Check\n")
    
    checks = [
        ("server", check_server_status),
        ("supabase", check_supabase_status),
        ("models", check_model_status),
        ("health", check_system_health)
    ]
    
    results = {}
    total_start = time.time()
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            results[check_name] = {
                "status": "error",
                "message": f"Check failed: {e}",
                "duration": 0
            }
        print()
    
    total_time = time.time() - total_start
    
    # Generate summary
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    warning_count = sum(1 for r in results.values() if r["status"] == "warning")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    
    print("📋 Health Check Summary:")
    print(f"   ✅ Healthy: {success_count}")
    print(f"   ⚠️ Warnings: {warning_count}")
    print(f"   ❌ Issues: {error_count}")
    print(f"   ⏱️ Total Time: {total_time:.2f}s")
    print()
    
    # Show detailed results
    for check_name, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "warning" else "❌"
        print(f"   {status_icon} {check_name.upper()}: {result['message']} ({result['duration']:.2f}s)")
    
    print()
    
    if error_count == 0:
        if warning_count == 0:
            print("🎉 All systems are healthy!")
            return True
        else:
            print("⚠️ System has warnings but no critical issues.")
            return True
    else:
        print(f"❌ {error_count} critical issue(s) found. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
