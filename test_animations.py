#!/usr/bin/env python3
"""
Test script to demonstrate the complete enhanced Space Mining game with polished adaptive UI.

Latest UI Polish Features:
1. Adaptive status panel with icon-based indicators in 2-column grid layout
2. Compact legend with visual icons matching game elements in 3-column layout
3. Horizontal event strip for recent actions instead of vertical text list
4. Visual icons replace text for faster recognition and better space utilization
5. All panels adapt to available horizontal space with optimized layouts

This script provides a complete demonstration of the polished space mining experience.
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the complete enhanced space mining game with polished adaptive UI."""
    print("🚀 Space Mining - Polished Adaptive UI Demo")
    print("=" * 75)
    print("🎨 UI POLISH & READABILITY FEATURES:")
    print("  📊 Adaptive Status Panel - 2-column grid with visual icons")
    print("  🎯 Icon-Based Indicators - Battery, package, pickaxe, clock, etc.")
    print("  📋 Compact Legend - 3-column layout with matching game icons")
    print("  ➡️ Horizontal Event Strip - Recent actions in compact format")
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
    print("=" * 75)
    print("🎮 Watch the polished adaptive UI in action!")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🎮 Environment initialized. Starting polished UI demo...")
        print()
        print("🔍 FOCUS ON NEW UI FEATURES:")
        print("  📊 STATUS PANEL (top-left):")
        print("    • 2-column grid layout for efficient space use")
        print("    • Battery icon shows energy level with visual fill")
        print("    • Package icon for inventory with cross-pattern")
        print("    • Pickaxe icon for total resources mined")
        print("    • Warning icon (!) for collisions with dynamic color")
        print("    • Clock icon for step counter")
        print("    • Asteroid icon for remaining asteroids")
        print()
        print("  📋 LEGEND (bottom-right):")
        print("    • 3-column compact grid layout")
        print("    • Visual icons that exactly match game elements")
        print("    • Agent circle, mothership glow, asteroid shape")
        print("    • Range circles, beam lines, particle dots")
        print("    • Much faster to scan than text-heavy legends")
        print()
        print("  ➡️ EVENT STRIP (below status):")
        print("    • Horizontal compact format for recent events")
        print("    • Shows mining results, deliveries, warnings")
        print("    • Pipes (|) separate different event types")
        print()
        print("  ⭐ Also watch for: starfield, mothership aura, all animations")
        print()
        
        ui_demo_messages = [
            "📊 Notice the compact 2-column status grid with visual icons",
            "🎯 Icons provide instant recognition - battery, package, pickaxe, etc.", 
            "📋 Legend uses 3-column layout with matching visual elements",
            "➡️ Recent events appear in horizontal strip format",
            "📐 All panels make efficient use of available screen space",
            "⚡ Visual icons are much faster to scan than text labels"
        ]
        
        message_index = 0
        
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
                print(f"📦 Carrying {agent_inventory:.1f} → mothership (watch package icon & event strip!)")
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
                        print(f"⛏️ Mining {nearest_asteroid[2]:.1f} resources (see pickaxe icon & score popup!)")
                    else:
                        # Move towards asteroid
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.6
                        print(f"🔍 Moving towards asteroid (distance: {min_distance:.1f})")
            
            # Show UI demo messages periodically
            if step_count % 200 == 0 and message_index < len(ui_demo_messages):
                print(f"💡 UI TIP: {ui_demo_messages[message_index]}")
                message_index += 1
            
            # Add some movement for starfield demo
            if step_count % 100 == 0:
                action[:2] += np.random.uniform(-0.4, 0.4, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight UI-relevant events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                print(f"✅ MINED {extracted:.1f}! (See pickaxe icon update & event strip shows '+{extracted:.1f}')")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 DELIVERED {delivered:.1f}! (Package icon resets, event strip shows delivery)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Warning icon flashes red, collision counter updates)")
            
            # Energy status for UI demo
            if agent_energy < 30 and step_count % 50 == 0:
                print(f"⚠️ LOW ENERGY! (Battery icon shows red, visual fill decreases)")
            
            # UI status updates
            if step_count % 150 == 0:
                print(f"📊 UI Demo Status - Step {step_count}:")
                print(f"   🔋 Battery Icon: Shows {agent_energy:.0f}/150 energy")
                print(f"   📦 Package Icon: Shows {agent_inventory:.0f} inventory")
                print(f"   ⛏️ Pickaxe Icon: Shows {cumulative_mining:.1f} total mined")
                print(f"   ⚠️ Warning Icon: Shows {env.collision_count} collisions")
                print(f"   🕒 Clock Icon: Shows {env.steps_count}/{env.max_episode_steps}")
                print(f"   🌕 Asteroid Icon: Shows active/total asteroids")
                print(f"   📋 Legend: 3-col layout with visual game element icons")
                print(f"   ➡️ Event Strip: Horizontal format for recent actions")
                
                active_asteroids = np.sum(env.asteroid_resources >= 0.1)
                print(f"   📐 Space Usage: Optimized 2-col status, 3-col legend")
                print(f"   ⚡ Recognition: Visual icons for instant understanding")
            
            # Check for end conditions
            if terminated or truncated:
                print()
                print("🎬 GAME OVER - UI DEMO COMPLETE!")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 SUCCESS: All asteroids depleted!")
                    else:
                        print("💥 FAILURE: Energy depleted or too many collisions!")
                if truncated:
                    print("⏰ TIME LIMIT: Episode truncated!")
                
                print()
                print("🎨 POLISHED UI FEATURES DEMONSTRATED:")
                print("  ✅ Adaptive status panel with 2-column icon grid")
                print("  ✅ Visual icons for instant recognition (battery, package, etc.)")
                print("  ✅ Compact 3-column legend with matching game icons")
                print("  ✅ Horizontal event strip for efficient space use")
                print("  ✅ Optimized layouts that adapt to available space")
                print("  ✅ Faster scanning and better readability")
                print()
                print("⏳ Waiting for game over screen to show final stats...")
                
                # Let the game over screen fully appear
                for _ in range(100):
                    env.render()
                    time.sleep(0.033)
                
                print("✅ Complete UI polish demo finished!")
                break
            
            # Run for demonstration duration
            if step_count > 1000:
                print()
                print("🎬 Polished UI demo completed successfully!")
                print()
                print("🏆 UI POLISH ACHIEVEMENTS:")
                print("  ✅ Replaced text-heavy status with visual icon grid")
                print("  ✅ Created adaptive 2-column status layout")
                print("  ✅ Designed 3-column legend with game-matching icons")
                print("  ✅ Implemented horizontal event strip format")
                print("  ✅ Optimized space usage across all UI panels")
                print("  ✅ Achieved faster recognition through visual elements")
                print("  ✅ Maintained all functionality while improving UX")
                print()
                print("🌟 Space Mining UI is now polished and user-friendly!")
                break
                
            # Optimal viewing speed
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        print("🎨 Polished adaptive UI successfully demonstrated!")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        env.close()
        print("🚪 Enhanced Space Mining UI demo complete!")
        print("✨ Thank you for experiencing the polished interface!")

if __name__ == "__main__":
    main()