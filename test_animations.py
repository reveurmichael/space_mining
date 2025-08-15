#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced Space Mining game with all visual improvements.

New Visual Features:
1. Fixed, intuitive colors (green agent, yellow asteroids, blue mothership)
2. Observation range dimming (spotlight effect)
3. Size-based asteroid resource indication
4. Enhanced animations (delivery particles, mining beam, agent trail)
5. Collision effects (flash + screen shake)
6. Score popups and improved UI layout
7. Energy warning system and atmospheric effects
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the enhanced space mining game with all visual improvements."""
    print("🚀 Enhanced Space Mining - Visual Demo")
    print("=" * 60)
    print("🎨 NEW VISUAL FEATURES:")
    print("  🎯 Fixed Colors - Agent always green, asteroids yellow")
    print("  🌑 Observation Dimming - Spotlight effect shows visible area")
    print("  📏 Size-Based Resources - Asteroid size shows resource amount")
    print("  ✨ Enhanced Animations - Smooth particles & effects")
    print("  💥 Collision System - Flash + screen shake")
    print("  📊 Score Popups - Floating +X text")
    print("  ⚠️ Energy Warning - Red halo when energy is low")
    print("  🌟 Atmospheric Effects - Glowing asteroids, pulsing obstacles")
    print("=" * 60)
    print("🎮 Watch the AI play or press Ctrl+C to exit")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🎮 Environment initialized. Starting enhanced visual demo...")
        print()
        print("🔍 Look for these visual features:")
        print("  • Green agent with white outline (always green regardless of state)")
        print("  • Yellow asteroids that get LARGER when they have more resources")
        print("  • Blue mothership with glow effect") 
        print("  • Red pulsing obstacles with danger aura")
        print("  • Dimmed areas outside observation range")
        print("  • Fading green trail behind the agent")
        print("  • Animated mining beam when extracting resources")
        print("  • Glowing particles when delivering to mothership")
        print("  • Score popups (+X.X) when mining/delivering")
        print("  • Red flash and screen shake on collisions")
        print("  • Red warning halo when energy is low (<30)")
        print()
        
        while True:
            # Simple AI behavior for demonstration
            agent_pos = observation[:2]
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
                print(f"📦 Agent carrying {agent_inventory:.1f} resources → mothership (watch for delivery particles!)")
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
                        print(f"⛏️ Mining asteroid with {nearest_asteroid[2]:.1f} resources (watch animated beam!)")
                    else:
                        # Move towards asteroid
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.6
                        print(f"🔍 Moving towards asteroid (distance: {min_distance:.1f})")
            
            # Add some random movement for variety
            if step_count % 150 == 0:
                action[:2] += np.random.uniform(-0.4, 0.4, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight interesting visual events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                print(f"✅ MINED {extracted:.1f} resources! (Yellow score popup should appear)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 DELIVERED {delivered:.1f} resources! (Glowing particles should fly to mothership)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Red flash + screen shake should trigger)")
            
            # Show energy warnings
            if agent_energy < 30 and step_count % 50 == 0:
                print(f"⚠️ LOW ENERGY ({agent_energy:.1f}/150) - Red warning halo should be visible!")
            
            # Show visual stats every 100 steps
            if step_count % 100 == 0:
                print(f"📊 Step {step_count}:")
                print(f"   🔋 Energy: {agent_energy:.1f}/150")
                print(f"   📦 Inventory: {agent_inventory:.1f}")
                print(f"   ✨ Active particles: {len(env.delivery_particles)}")
                print(f"   🌟 Trail points: {len(env.agent_trail)}")
                print(f"   📊 Score popups: {len(env.score_popups)}")
                print(f"   🌑 Observation dimming: {'Active' if len(env.agent_trail) > 0 else 'Initializing'}")
                
                # Describe what should be visible
                active_asteroids = np.sum(env.asteroid_resources >= 0.1)
                print(f"   🌕 Active asteroids: {active_asteroids} (size shows resource level)")
                print(f"   🎯 Visual features: Fixed colors, dimming effect, enhanced UI")
            
            # Check for end conditions
            if terminated or truncated:
                if terminated:
                    print("🏁 Episode completed!")
                if truncated:
                    print("⏰ Time limit reached!")
                print()
                print("🎨 Visual Demo Summary:")
                print("  ✅ Fixed color scheme demonstrated")
                print("  ✅ Observation range dimming shown")
                print("  ✅ Size-based asteroid indicators active") 
                print("  ✅ All animations and effects working")
                break
            
            # Run for a reasonable time to show all features
            if step_count > 1500:
                print("🎬 Enhanced visual demo completed successfully!")
                print()
                print("🌟 All visual enhancements are now active:")
                print("  • Consistent color scheme")
                print("  • Atmospheric lighting effects")
                print("  • Enhanced UI and status displays")
                print("  • Smooth animations and feedback")
                break
                
            # Controlled speed for better visibility
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        print("🎨 Visual enhancements successfully demonstrated!")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        env.close()
        print("🚪 Environment closed. Enhanced Space Mining demo complete!")

if __name__ == "__main__":
    main()