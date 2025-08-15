#!/usr/bin/env python3
"""
Test script to demonstrate the new visual enhancements in the Space Mining game.

This script will run the game and showcase:
1. Resource delivery animation (glowing dots)
2. Collision impact flash and screen shake
3. Agent trail fade
4. Animated mining beam 
5. Score popups

Press ESC to quit, or let it run to see all the animations.
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the enhanced space mining game with all animations."""
    print("🚀 Starting Space Mining with Enhanced Animations!")
    print("=" * 50)
    print("New Features:")
    print("🌟 Resource delivery particles - glowing dots travel to mothership")
    print("💥 Collision flash - red screen flash and shake on collision")
    print("✨ Agent trail - fading comet-like trail showing path")
    print("⚡ Animated mining beam - scrolling dashed energy beam")
    print("📊 Score popups - floating +X text when mining/delivering")
    print("🎨 Enhanced visuals - better colors, emojis, and polish")
    print("=" * 50)
    print("Controls: Use arrow keys or WASD to move, SPACE to mine")
    print("Press ESC to quit, or watch the AI play automatically")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🎮 Environment reset. Starting gameplay...")
        
        while True:
            # Simple AI behavior for demonstration
            agent_pos = observation[:2]
            agent_vel = observation[2:4]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Simple AI logic
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership if carrying resources
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Agent carrying {agent_inventory:.1f} resources, heading to mothership")
            else:
                # Find nearest asteroid
                nearest_asteroid = None
                min_distance = float('inf')
                
                for i, asteroid in enumerate(asteroid_obs):
                    if np.linalg.norm(asteroid[:2]) > 0.1:  # Valid asteroid
                        distance = np.linalg.norm(asteroid[:2])
                        if distance < min_distance:
                            min_distance = distance
                            nearest_asteroid = asteroid
                
                if nearest_asteroid is not None:
                    if min_distance < env.mining_range:
                        # Mine if close enough
                        action[2] = 1.0  # Mine
                        print(f"⛏️ Mining asteroid with {nearest_asteroid[2]:.1f} resources")
                    else:
                        # Move towards asteroid
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.6
                        print(f"🔍 Moving towards asteroid (distance: {min_distance:.1f})")
            
            # Add some random movement for demonstration
            if step_count % 200 == 0:
                action[:2] += np.random.uniform(-0.3, 0.3, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print interesting events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                print(f"✅ Mined {extracted:.1f} resources! (Score popup should appear)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 Delivered {delivered:.1f} resources! (Particle animation should play)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Flash and shake effects should trigger)")
            
            # Show status every 100 steps
            if step_count % 100 == 0:
                print(f"📊 Step {step_count}: Energy={agent_energy:.1f}, Inventory={agent_inventory:.1f}")
                print(f"   Particles active: {len(env.delivery_particles)}")
                print(f"   Trail points: {len(env.agent_trail)}")
                print(f"   Score popups: {len(env.score_popups)}")
            
            # Check for end conditions
            if terminated or truncated:
                if terminated:
                    print("🏁 Episode terminated!")
                if truncated:
                    print("⏰ Episode truncated (time limit reached)")
                break
            
            # Run for a reasonable time to show animations
            if step_count > 2000:
                print("🎬 Animation demo completed!")
                break
                
            # Small delay to make it easier to see
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n👋 Game interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        env.close()
        print("🚪 Environment closed. Thanks for watching the animation demo!")

if __name__ == "__main__":
    main()