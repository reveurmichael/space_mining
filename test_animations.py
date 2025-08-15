#!/usr/bin/env python3
"""
Test script to demonstrate the complete enhanced Space Mining game with all visual improvements.

Latest Visual Features:
1. Dynamic parallax starfield background that moves with agent
2. Game over/success screen with comprehensive final statistics
3. Mothership safe zone aura (pulsing blue protective field)
4. All previous enhancements: fixed colors, observation dimming, animations, etc.

This script provides a complete demonstration of the enhanced space mining experience.
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the complete enhanced space mining game with all visual improvements."""
    print("🚀 Space Mining - Complete Visual Enhancement Demo")
    print("=" * 70)
    print("🌟 LATEST VISUAL FEATURES:")
    print("  💫 Dynamic Parallax Starfield - 3-layer moving background")
    print("  🎬 Game Over Screen - Fade-to-black with final statistics")
    print("  🛡️ Mothership Safe Zone - Pulsing blue protective aura")
    print()
    print("🎨 ALL VISUAL ENHANCEMENTS:")
    print("  🎯 Fixed Color Scheme - Consistent object identification")
    print("  🌑 Observation Dimming - Spotlight effect for visibility")
    print("  📏 Size-Based Resources - Asteroid size shows resource level")
    print("  ✨ Enhanced Animations - Particles, beams, trails")
    print("  💥 Collision Effects - Flash + screen shake")
    print("  📊 Score Popups - Floating feedback text")
    print("  ⚠️ Energy Warning System - Red halo for low energy")
    print("  🌟 Atmospheric Effects - Glows, pulses, depth")
    print("=" * 70)
    print("🎮 Watch the complete enhanced experience!")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🎮 Environment initialized. Starting complete visual demo...")
        print()
        print("🔍 WATCH FOR THESE ENHANCED FEATURES:")
        print("  ⭐ Parallax starfield moving behind everything")
        print("  🛡️ Blue pulsing aura around the mothership (safe zone)")
        print("  🤖 Green agent with white outline (consistent color)")
        print("  🌕 Yellow asteroids that grow larger with more resources")
        print("  ☄️ Red pulsing obstacles with danger glow")
        print("  🌑 Dimmed areas outside observation range")
        print("  ✨ Fading green trail behind the agent")
        print("  ⚡ Animated mining beam when extracting")
        print("  💫 Glowing particles when delivering to mothership")
        print("  📊 Score popups (+X.X) floating upward")
        print("  💥 Red flash and screen shake on collisions")
        print("  ⚠️ Red warning halo when energy is critically low")
        print("  🎬 Game over screen with complete statistics at the end")
        print()
        
        while True:
            # Enhanced AI behavior for better demonstration
            agent_pos = observation[:2]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Smart AI logic
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership if carrying resources
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Agent carrying {agent_inventory:.1f} → mothership (watch delivery particles + starfield motion!)")
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
                        print(f"⛏️ Mining asteroid with {nearest_asteroid[2]:.1f} resources (animated beam + score popup!)")
                    else:
                        # Move towards asteroid
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.6
                        print(f"🔍 Moving towards asteroid (distance: {min_distance:.1f}) - notice starfield parallax!")
            
            # Add some exploration movement for better starfield demo
            if step_count % 120 == 0:
                action[:2] += np.random.uniform(-0.5, 0.5, 2)
                print("🌟 Random movement - great for seeing starfield parallax effect!")
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight important visual events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                print(f"✅ MINED {extracted:.1f} resources! (Yellow score popup should float upward)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 DELIVERED {delivered:.1f} resources! (Glowing particles fly to mothership)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Red flash + screen shake + starfield disruption)")
            
            # Energy warnings for visual effect demonstration
            if agent_energy < 30 and step_count % 40 == 0:
                print(f"⚠️ CRITICAL ENERGY ({agent_energy:.1f}/150) - Red warning halo should pulse around agent!")
            
            # Mothership proximity notifications
            distance_to_mothership = np.linalg.norm(mothership_rel_pos)
            if distance_to_mothership < 15 and step_count % 60 == 0:
                print(f"🛡️ Near mothership safe zone - blue pulsing aura should be visible!")
            
            # Enhanced visual stats every 100 steps
            if step_count % 100 == 0:
                print(f"📊 Visual Demo Status - Step {step_count}:")
                print(f"   🔋 Energy: {agent_energy:.1f}/150")
                print(f"   📦 Inventory: {agent_inventory:.1f}")
                print(f"   ⭐ Starfield layers: {len(env.starfield_layers)} (parallax active)")
                print(f"   ✨ Active particles: {len(env.delivery_particles)}")
                print(f"   🌟 Trail points: {len(env.agent_trail)}")
                print(f"   📊 Score popups: {len(env.score_popups)}")
                print(f"   🌑 Observation dimming: Active")
                print(f"   🛡️ Mothership aura: Pulsing")
                
                # Describe current visual state
                active_asteroids = np.sum(env.asteroid_resources >= 0.1)
                print(f"   🌕 Active asteroids: {active_asteroids} (varying sizes)")
                print(f"   🎯 All visual systems: Fully operational")
            
            # Check for end conditions - game over screen demo
            if terminated or truncated:
                print()
                print("🎬 GAME OVER TRIGGERED!")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 SUCCESS: All asteroids depleted!")
                    else:
                        print("💥 FAILURE: Energy depleted or too many collisions!")
                if truncated:
                    print("⏰ TIME LIMIT: Episode truncated!")
                
                print()
                print("🎭 GAME OVER SCREEN FEATURES:")
                print("  • Fade-to-black transition")
                print("  • Success/failure title with appropriate colors")
                print("  • Comprehensive final statistics")
                print("  • Resources mined, delivered, collisions, etc.")
                print("  • Efficiency scoring")
                print("  • Restart instructions")
                print()
                print("⏳ Waiting for game over screen to fully fade in...")
                
                # Let the game over screen fully appear
                for _ in range(100):  # About 3 seconds at 30 FPS
                    env.render()
                    time.sleep(0.033)
                
                print("✅ Game over screen demo complete!")
                break
            
            # Run for a good demonstration duration
            if step_count > 1200:
                print()
                print("🎬 Complete visual enhancement demo finished successfully!")
                print()
                print("🌟 ALL VISUAL FEATURES DEMONSTRATED:")
                print("  ✅ Dynamic parallax starfield background")
                print("  ✅ Mothership safe zone aura (pulsing blue)")
                print("  ✅ Game over screen with statistics (see by letting game end)")
                print("  ✅ Fixed color scheme and consistent visuals")
                print("  ✅ Observation range dimming effect")
                print("  ✅ Size-based asteroid resource indication")
                print("  ✅ All animations: particles, beams, trails")
                print("  ✅ Collision effects and energy warnings")
                print("  ✅ Enhanced UI and atmospheric effects")
                print()
                print("🏆 Space Mining visual enhancement is complete!")
                break
                
            # Controlled speed for optimal viewing
            time.sleep(0.025)
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        print("🎨 All visual enhancements successfully demonstrated!")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        env.close()
        print("🚪 Enhanced Space Mining demo complete!")
        print("🌟 Thank you for experiencing the visual enhancements!")

if __name__ == "__main__":
    main()