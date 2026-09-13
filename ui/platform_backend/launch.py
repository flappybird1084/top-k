"""Run the existing engine with a reservation that fits this job's spend cap."""
import os
import sys
from pathlib import Path

engine=Path(os.environ.get('TOPK_ENGINE_ROOT','/marimo/top-k'))
sys.path.insert(0,str(engine))
import search

original_config=search.active_config

def platform_config():
    config=original_config()
    # Reservations are API-equivalent estimates, not a provider-side hard spending limit.
    config['max_call_reservation_usd']=min(config['max_call_reservation_usd'],float(os.environ['TOPK_CALL_RESERVATION']))
    return config

search.active_config=platform_config
if __name__=='__main__':search.main()
