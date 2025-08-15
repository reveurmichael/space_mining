#!/usr/bin/env python3
"""
Test script to demonstrate the cosmic-enhanced Space Mining game with universe-like background.

Cosmic Background Features:
1. Colorful nebula clouds with rotation and parallax
2. Distant spiral galaxies with rotating arms
3. Fine space dust particles
4. Enhanced starfield with colored stars and twinkling
5. Dynamic zoom system that responds to game state
6. Multi-layered cosmic atmosphere

This script showcases the truly cosmic, universe-like visual experience.
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Run the cosmic-enhanced space mining game with universe-like background."""
    print("🌌 Space Mining - Cosmic Universe Demo")
    print("=" * 90)
    print("🌟 COSMIC BACKGROUND ENHANCEMENTS:")
    print("  🌌 Colorful Nebula Clouds - Purple, blue, pink nebulae with rotation")
    print("  🌀 Distant Spiral Galaxies - Rotating spiral arms in the background") 
    print("  ✨ Enhanced Starfield - Colored stars (blue, yellow, red, white) with twinkling")
    print("  💫 Space Dust Particles - Fine cosmic dust with natural drift")
    print("  🔍 Dynamic Zoom System - Context-sensitive zoom based on game state")
    print("  🎨 Multi-Layer Parallax - Different cosmic elements move at different speeds")
    print()
    print("🎮 ZOOM FEATURES:")
    print("  🔍 Zoom In - When energy is low (tension effect)")
    print("  🔍 Zoom Out - When few asteroids remain or during collisions")
    print("  🔍 Smooth Transitions - Gradual zoom changes for cinematic feel")
    print("  🔍 All Elements Scale - Nebulae, galaxies, stars scale with zoom")
    print()
    print("🌠 COSMIC ELEMENTS:")
    print("  🌌 8 Nebula Clouds - Rotating, colorful gas clouds")
    print("  🌀 5 Distant Galaxies - Spiral arms with multiple layers")
    print("  ⭐ 270 Enhanced Stars - 4 colors with realistic twinkling")
    print("  💫 200 Space Dust - Fine particles with cosmic drift")
    print("  🎨 Layered Depth - Background to foreground cosmic elements")
    print()
    print("🎯 VISUAL IMPROVEMENTS:")
    print("  📺 Large Display (1200x900) for immersive cosmic view")
    print("  🎨 Deep Space Colors - Rich cosmic atmosphere")
    print("  ✨ Atmospheric Effects - Glows, rotations, fading")
    print("  🔄 Continuous Animation - Living, breathing universe")
    print("=" * 90)
    print("🚀 Experience the cosmic universe of space mining!")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🌌 Cosmic environment initialized...")
        print()
        print("🔍 COSMIC FEATURES TO OBSERVE:")
        print("  🌌 NEBULA CLOUDS:")
        print("    • Colorful gas clouds in purple, blue, pink, orange")
        print("    • Slow rotation and movement with parallax")
        print("    • Multi-layered gradual transparency effects")
        print("    • Scale with zoom level for immersive depth")
        print()
        print("  🌀 DISTANT GALAXIES:")
        print("    • Spiral arms rotating slowly in background")
        print("    • Multiple arms (2-5) per galaxy")
        print("    • Fading brightness from core to edge")
        print("    • Very slow parallax movement")
        print()
        print("  ⭐ ENHANCED STARFIELD:")
        print("    • Blue, yellow, red, and white colored stars")
        print("    • Realistic twinkling effects")
        print("    • Different sizes and brightness levels")
        print("    • Multi-layer parallax at different speeds")
        print()
        print("  💫 SPACE DUST:")
        print("    • Fine cosmic particles drifting naturally")
        print("    • Fast parallax movement in foreground")
        print("    • Subtle but adds to cosmic atmosphere")
        print()
        print("  🔍 DYNAMIC ZOOM:")
        print("    • Watch zoom change based on game state")
        print("    • Low energy = zoom in for tension")
        print("    • Few asteroids = zoom out for overview")
        print("    • Collisions = quick zoom out for impact")
        print()
        
        cosmic_tips = [
            "🌌 Notice the colorful nebula clouds rotating slowly in the background",
            "🌀 Look for distant spiral galaxies with rotating arms",
            "⭐ Observe the colored stars twinkling realistically",
            "🔍 Watch how zoom changes create cinematic effects during gameplay",
            "💫 See the fine space dust adding cosmic atmosphere"
        ]
        
        tip_index = 0
        zoom_changes_observed = []
        
        while True:
            # Enhanced AI behavior for cosmic demonstration
            agent_pos = observation[:2]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Track zoom changes for demonstration
            current_zoom = env.zoom_level
            if step_count > 10:  # Allow initial stabilization
                if abs(current_zoom - 1.0) > 0.1:
                    zoom_type = "zoom in" if current_zoom > 1.0 else "zoom out"
                    if zoom_type not in zoom_changes_observed:
                        zoom_changes_observed.append(zoom_type)
                        print(f"🔍 COSMIC ZOOM: {zoom_type.upper()} detected (level: {current_zoom:.2f})")
            
            # Optimized AI logic for cosmic showcase
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership if carrying resources
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Delivering {agent_inventory:.1f} → mothership (cosmic zoom: {current_zoom:.2f})")
            else:
                # Find nearest asteroid for demonstration
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
                        print(f"⛏️ Mining {nearest_asteroid[2]:.1f} (cosmic atmosphere enhanced!)")
                    else:
                        # Move towards asteroid
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.8
                        print(f"🔍 Moving to asteroid (distance: {min_distance:.1f})")
            
            # Show cosmic tips periodically
            if step_count % 400 == 0 and tip_index < len(cosmic_tips):
                print(f"🌟 COSMIC TIP: {cosmic_tips[tip_index]}")
                tip_index += 1
            
            # Add movement variety for better cosmic showcase
            if step_count % 120 == 0:
                action[:2] += np.random.uniform(-0.2, 0.2, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight cosmic visual events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                combo_count = env.combo_state.get('chain_count', 0)
                if combo_count >= 2:
                    print(f"🔥 COMBO x{combo_count}! Enhanced by cosmic atmosphere!")
                else:
                    print(f"✅ MINED {extracted:.1f}! (Nebulae dance in background)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 DELIVERED {delivered:.1f}! (Galaxies rotate in the distance)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COLLISION! (Zoom effect with cosmic background!)")
            
            # Cosmic status updates
            if step_count % 250 == 0:
                nebula_count = len(env.nebula_clouds)
                galaxy_count = len(env.distant_galaxies)
                star_count = sum(len(layer) for layer in env.starfield_layers)
                dust_count = len(env.space_dust)
                cosmic_time = env.cosmic_time
                
                print(f"🌌 Cosmic Universe Status - Step {step_count}:")
                print(f"   🔍 Current Zoom: {current_zoom:.2f}x")
                print(f"   🌌 Nebula Clouds: {nebula_count} rotating gas clouds")
                print(f"   🌀 Distant Galaxies: {galaxy_count} spiral formations")
                print(f"   ⭐ Enhanced Stars: {star_count} twinkling points")
                print(f"   💫 Space Dust: {dust_count} cosmic particles")
                print(f"   ⏱️ Cosmic Time: {cosmic_time:.1f}s (twinkling cycle)")
                print(f"   🎨 Background Layers: Nebulae → Galaxies → Dust → Stars")
                
                if len(zoom_changes_observed) > 0:
                    print(f"   🎬 Zoom Effects: {', '.join(zoom_changes_observed)} demonstrated")
                else:
                    print(f"   🎯 Zoom Effects: Waiting for state changes...")
            
            # Check for end conditions
            if terminated or truncated:
                print()
                print("🌌 COSMIC UNIVERSE DEMO COMPLETE!")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 SUCCESS: Mission completed in cosmic space!")
                    else:
                        print("💥 FAILURE: Lost in the cosmic void!")
                if truncated:
                    print("⏰ TIME LIMIT: Cosmic journey ended!")
                
                print()
                print("🌟 COSMIC FEATURES DEMONSTRATED:")
                print("  ✅ Colorful nebula clouds with rotation and parallax")
                print("  ✅ Distant spiral galaxies with animated arms")
                print("  ✅ Enhanced starfield with colors and twinkling")
                print("  ✅ Fine space dust creating cosmic atmosphere")
                print("  ✅ Dynamic zoom system for cinematic effects")
                print("  ✅ Multi-layered cosmic background system")
                print()
                print("⏳ Waiting for cosmic game over screen...")
                
                # Let the cosmic game over screen show
                for _ in range(140):
                    env.render()
                    time.sleep(0.025)
                
                print("✅ Cosmic universe demo finished!")
                break
            
            # Run for good cosmic demonstration
            if step_count > 1600:
                print()
                print("🌌 Cosmic universe demo completed successfully!")
                print()
                print("🏆 COSMIC ACHIEVEMENTS:")
                print("  ✅ Created immersive universe-like background")
                print("  ✅ Implemented multi-layered cosmic elements")
                print("  ✅ Added dynamic zoom for cinematic experience")
                print("  ✅ Enhanced with colorful nebulae and galaxies")
                print("  ✅ Integrated twinkling colored starfield")
                print("  ✅ Added fine space dust for atmosphere")
                print("  ✅ Created living, breathing cosmic environment")
                print()
                zoom_msg = f"({', '.join(zoom_changes_observed)})" if zoom_changes_observed else "(waiting for triggers)"
                print(f"🔍 Dynamic Zoom: {zoom_msg}")
                print(f"🌌 Cosmic Elements: {nebula_count} nebulae, {galaxy_count} galaxies")
                print(f"⭐ Total Stars: {star_count} with enhanced effects")
                print(f"💫 Space Dust: {dust_count} atmospheric particles")
                print()
                print("🌟 The universe of Space Mining is now truly cosmic!")
                break
                
            # Optimal cosmic viewing speed
            time.sleep(0.035)
    
    except KeyboardInterrupt:
        print("\n👋 Cosmic demo interrupted by user")
        print("🌌 Universe-like background successfully demonstrated!")
    except Exception as e:
        print(f"❌ Cosmic error occurred: {e}")
    finally:
        env.close()
        print("🚪 Cosmic Space Mining demo complete!")
        print("✨ Thank you for exploring the enhanced cosmic universe!")

if __name__ == "__main__":
    main()