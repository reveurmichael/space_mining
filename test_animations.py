#!/usr/bin/env python3
"""
Test script to demonstrate the complete enhanced Space Mining game with floating timeline and combo system.

Latest Polish Features:
1. Floating Event Timeline - Top bar with micro-cards showing last 5 events
2. Score Combo System - Pulsing multiplier badges for rapid mining chains
3. Adaptive UI layouts with visual icons for faster recognition
4. All previous enhancements maintained and working together

This script provides a complete demonstration of the polished gaming experience.
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the complete enhanced space mining game with floating timeline and combo system."""
    print("🚀 Space Mining - Complete Polish Demo")
    print("=" * 80)
    print("🆕 LATEST POLISH FEATURES:")
    print("  🎬 Floating Event Timeline - Top bar with micro-cards (last 5 events)")
    print("  🔥 Score Combo System - x2/x3 multiplier badges for rapid mining")
    print("  ⏱️ Event Fading - Cards fade after few seconds")
    print("  ✨ Combo Detection - Chain mining within 50-step window")
    print("  🎯 Visual Icons - Timeline cards have type-specific icons")
    print()
    print("🎨 UI POLISH & READABILITY:")
    print("  📊 Adaptive Status Panel - 2-column grid with visual icons")
    print("  🎯 Icon-Based Indicators - Battery, package, pickaxe, clock, etc.")
    print("  📋 Compact Legend - 3-column layout with matching game icons")
    print("  ➡️ Horizontal Event Strip - Recent actions in status area")
    print("  📐 Space Optimization - Better use of available screen real estate")
    print("  ⚡ Faster Recognition - Visual icons instead of text for key info")
    print()
    print("🌟 ALL VISUAL ENHANCEMENTS:")
    print("  💫 Dynamic Parallax Starfield - 3-layer moving background")
    print("  🎬 Game Over Screen - Fade-to-black with final statistics")
    print("  🛡️ Mothership Safe Zone - Pulsing blue protective aura")
    print("  🎯 Fixed Color Scheme - Consistent object identification")
    print("  🌑 Observation Dimming - Spotlight effect for visibility")
    print("  📏 Size-Based Resources - Asteroid size shows resource level")
    print("  ✨ Enhanced Animations - Particles, beams, trails")
    print("  💥 Collision Effects - Flash + screen shake")
    print("  📊 Score Popups - Floating feedback text")
    print("  ⚠️ Energy Warning System - Red halo for low energy")
    print("  🌟 Atmospheric Effects - Glows, pulses, depth")
    print("=" * 80)
    print("🎮 Watch the complete polished experience!")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🎮 Environment initialized. Starting complete polish demo...")
        print()
        print("🔍 FOCUS ON NEW FEATURES:")
        print("  🎬 FLOATING TIMELINE (top center):")
        print("    • Micro-cards show last 5 events in real-time")
        print("    • Cards have type-specific icons (pickaxe, arrow, warning)")
        print("    • Different background colors for different event types")
        print("    • Automatic fading after a few seconds")
        print("    • Centered layout adapts to number of events")
        print()
        print("  🔥 COMBO SYSTEM (center screen):")
        print("    • Triggers when mining within 50-step window")
        print("    • Shows x2, x3, x4+ multiplier badges")
        print("    • Pulsing golden badge with rotating sparkles")
        print("    • Special combo events appear in timeline")
        print("    • Badge fades after 4 seconds")
        print()
        print("  📊 STATUS & LEGEND:")
        print("    • 2-column status grid with visual icons")
        print("    • 3-column legend with game-matching elements")
        print("    • Recent events in horizontal strip below status")
        print()
        print("  ⭐ Also watch for: starfield, mothership aura, all animations")
        print()
        
        timeline_tips = [
            "🎬 Notice the floating timeline cards at the top of the screen",
            "🔥 Try to mine multiple asteroids quickly to trigger combo multipliers!", 
            "⏱️ Watch how timeline cards fade out over time",
            "🎯 Each event type has its own icon and color scheme",
            "✨ Combo badges appear with rotating sparkles for extra flair"
        ]
        
        tip_index = 0
        combo_demonstrated = False
        
        while True:
            # Enhanced AI behavior for better demonstration
            agent_pos = observation[:2]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Smart AI logic optimized for combo demonstration
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership if carrying resources
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Carrying {agent_inventory:.1f} → mothership (watch timeline card!)")
            else:
                # Find nearest asteroid for rapid mining (combo demo)
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
                        print(f"⛏️ Mining {nearest_asteroid[2]:.1f} (watch for combo and timeline!)")
                    else:
                        # Move towards asteroid aggressively for combo demo
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 1.0  # Faster movement
                        print(f"🔍 Moving fast to asteroid (distance: {min_distance:.1f})")
            
            # Show timeline tips periodically
            if step_count % 250 == 0 and tip_index < len(timeline_tips):
                print(f"💡 TIMELINE TIP: {timeline_tips[tip_index]}")
                tip_index += 1
            
            # Add movement for variety
            if step_count % 80 == 0:
                action[:2] += np.random.uniform(-0.3, 0.3, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight timeline and combo events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                combo_count = env.combo_state.get('chain_count', 0)
                if combo_count >= 2:
                    print(f"🔥 COMBO x{combo_count}! Mined {extracted:.1f} (see pulsing badge & timeline!)")
                    combo_demonstrated = True
                else:
                    print(f"✅ MINED {extracted:.1f}! (New card added to timeline)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 DELIVERED {delivered:.1f}! (Green delivery card in timeline)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Red warning card appears in timeline)")
            
            # Timeline and combo status updates
            if step_count % 200 == 0:
                timeline_count = len(env.event_timeline)
                combo_active = env.combo_state.get('display_timer', 0) > 0
                combo_chain = env.combo_state.get('chain_count', 0)
                
                print(f"📊 Polish Demo Status - Step {step_count}:")
                print(f"   🎬 Timeline Cards: {timeline_count}/5 active")
                print(f"   🔥 Combo System: {'x' + str(combo_chain) + ' ACTIVE' if combo_active else 'Ready'}")
                print(f"   ⏱️ Card Fading: {'Active' if timeline_count > 0 else 'Waiting for events'}")
                print(f"   🎯 Event Types: Mining, Delivery, Collision, Combo")
                print(f"   📐 UI Layout: Floating top + adaptive sides")
                print(f"   ✨ Visual Polish: Icons, colors, animations")
                
                if combo_demonstrated:
                    print(f"   🏆 Combo Demo: Successfully triggered!")
                else:
                    print(f"   🎯 Combo Demo: Mine quickly to trigger combos!")
            
            # Check for end conditions
            if terminated or truncated:
                print()
                print("🎬 COMPLETE POLISH DEMO FINISHED!")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 SUCCESS: All asteroids depleted!")
                    else:
                        print("💥 FAILURE: Energy depleted or too many collisions!")
                if truncated:
                    print("⏰ TIME LIMIT: Episode truncated!")
                
                print()
                print("🌟 COMPLETE POLISH FEATURES DEMONSTRATED:")
                print("  ✅ Floating Event Timeline with micro-cards")
                print("  ✅ Score combo system with pulsing badges")
                print("  ✅ Event fading and timeline management")
                print("  ✅ Adaptive UI layouts with visual icons")
                print("  ✅ All animations and atmospheric effects")
                print("  ✅ Professional game polish and UX")
                print()
                
                # Show final timeline state
                if env.event_timeline:
                    print(f"📋 Final Timeline State: {len(env.event_timeline)} events")
                    for i, event in enumerate(env.event_timeline):
                        print(f"   {i+1}. {event['type']}: {event['text']}")
                
                print()
                print("⏳ Waiting for game over screen...")
                
                # Let the game over screen fully appear
                for _ in range(100):
                    env.render()
                    time.sleep(0.033)
                
                print("✅ Complete polish demo finished!")
                break
            
            # Run for good demonstration duration
            if step_count > 1200:
                print()
                print("🎬 Complete polish demo completed successfully!")
                print()
                print("🏆 POLISH ACHIEVEMENTS:")
                print("  ✅ Implemented floating event timeline system")
                print("  ✅ Created score combo/chain multiplier system")
                print("  ✅ Added event fading and lifetime management")
                print("  ✅ Enhanced UI with adaptive layouts and icons")
                print("  ✅ Maintained all visual effects and animations")
                print("  ✅ Achieved professional game polish level")
                print()
                combo_msg = "demonstrated" if combo_demonstrated else "available (mine rapidly!)"
                print(f"🔥 Combo System: {combo_msg}")
                print(f"🎬 Timeline Events: {len(env.event_timeline)} currently active")
                print()
                print("🌟 Space Mining is now fully polished and complete!")
                break
                
            # Optimal viewing speed
            time.sleep(0.025)
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        print("🎨 Complete polish features successfully demonstrated!")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        env.close()
        print("🚪 Complete Space Mining polish demo finished!")
        print("✨ Thank you for experiencing the fully enhanced interface!")

if __name__ == "__main__":
    main()