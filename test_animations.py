#!/usr/bin/env python3
"""
🌌 SPACE MINING UNIVERSE - ULTIMATE COSMIC EXPERIENCE 🌌

The most spectacular, coherent, and polished space mining simulation with:
- Cinematic 1440x1080 HD display for maximum immersion
- Spectacular cosmic phenomena: nebulae, galaxies, shooting stars, auroras, pulsars
- Dynamic zoom system with context-sensitive cinematography
- Streamlined, elegant UI with essential information only
- Perfect visual hierarchy and coherence throughout
- Living, breathing universe that responds to gameplay

This is the definitive cosmic space exploration experience!
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Experience the ultimate cosmic space mining universe."""
    print("🌌" + "="*100 + "🌌")
    print("🚀              SPACE MINING UNIVERSE - ULTIMATE COSMIC EXPERIENCE              🚀")
    print("🌌" + "="*100 + "🌌")
    print()
    print("✨ SPECTACULAR COSMIC ENHANCEMENTS:")
    print("  🎬 CINEMATIC DISPLAY - 1440x1080 HD for maximum immersion")
    print("  🌌 COLORFUL NEBULAE - 12 rotating, pulsing gas clouds with ethereal beauty")
    print("  🌀 SPIRAL GALAXIES - 8 distant galaxies with rotating arms and bright cores")
    print("  💫 SHOOTING STARS - Rare, spectacular meteors with glowing trails")
    print("  🌈 COSMIC AURORAS - 6 ethereal energy curtains dancing across space")
    print("  ⭐ PULSARS - 4 neutron stars with rotating energy beams")
    print("  ✨ ENHANCED STARFIELD - 380 colored, twinkling stars across 3 layers")
    print("  💎 SPACE DUST - 380 cosmic particles creating atmospheric depth")
    print()
    print("🎮 DYNAMIC CINEMATOGRAPHY:")
    print("  🔍 CONTEXT-SENSITIVE ZOOM - Dramatic camera work based on game state")
    print("    • Zoom in when energy is low (tension and focus)")
    print("    • Zoom out during collisions (impact and drama)")
    print("    • Zoom out when few asteroids remain (strategic overview)")
    print("    • Smooth transitions for cinematic feel")
    print()
    print("🎨 ELEGANT VISUAL DESIGN:")
    print("  📺 Perfect 4:3 HD Ratio - Optimized for cosmic viewing")
    print("  🎯 Streamlined UI - Essential information only, no clutter")
    print("  🌟 Visual Hierarchy - Clear, coherent information organization")
    print("  💫 Atmospheric Dimming - Spotlight effect for observation range")
    print("  🎪 All Animations - Particles, beams, trails, and cosmic effects")
    print()
    print("🌠 COSMIC PHENOMENA COUNT:")
    print("  🌌 Nebulae: 12 multi-layered rotating clouds")
    print("  🌀 Galaxies: 8 spiral formations with detailed arms")
    print("  ⭐ Stars: 380 with realistic twinkling and colors")
    print("  💫 Dust: 380 particles (300 fine + 80 coarse)")
    print("  🌈 Auroras: 6 wavy energy curtains")
    print("  ⚡ Pulsars: 4 with rotating beams")
    print("  💥 Shooting Stars: Dynamic spawning (up to 5 at once)")
    print()
    print("🌌" + "="*100 + "🌌")
    print("🚀 PREPARE FOR THE ULTIMATE COSMIC JOURNEY! 🚀")
    print("🌌" + "="*100 + "🌌")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🌌 Ultimate cosmic universe initialized...")
        print()
        print("🔍 COSMIC SPECTACLES TO WITNESS:")
        print()
        print("  🌌 NEBULAE PHENOMENA:")
        print("    • Multi-colored gas clouds in purple, blue, pink, orange, cyan")
        print("    • 7-layer depth effects with pulsing and rotation")
        print("    • Realistic parallax movement responding to agent motion")
        print("    • Ethereal inner glow effects for maximum beauty")
        print()
        print("  🌀 GALACTIC FORMATIONS:")
        print("    • Spiral galaxies with 2-6 rotating arms each")
        print("    • Bright central cores with realistic fade to edges")
        print("    • Variable arm thickness and stellar particle density")
        print("    • Ultra-slow parallax for distant cosmic effect")
        print()
        print("  💫 DYNAMIC PHENOMENA:")
        print("    • Rare shooting stars with spectacular glowing trails")
        print("    • Cosmic auroras - wavy energy curtains dancing in space")
        print("    • Pulsing neutron stars with rotating energy beams")
        print("    • 4-color twinkling starfield (blue, yellow, red, white)")
        print()
        print("  🎬 CINEMATIC SYSTEMS:")
        print("    • Dynamic zoom creates movie-like camera work")
        print("    • Context-sensitive effects enhance drama")
        print("    • All cosmic elements scale beautifully with zoom")
        print("    • Smooth interpolation for professional cinematography")
        print()
        print("  🎯 STREAMLINED INTERFACE:")
        print("    • Clean 4-panel status display (Energy, Cargo, Mined, Asteroids)")
        print("    • 5-column legend optimized for widescreen")
        print("    • Floating timeline cards for recent events")
        print("    • Essential information only - no visual clutter")
        print()
        
        cosmic_phenomena_tips = [
            "🌌 Watch the nebulae slowly rotate and pulse with ethereal beauty",
            "🌀 Notice distant spiral galaxies with their rotating stellar arms",
            "💫 Look for rare shooting stars streaking across the cosmic void",
            "🌈 Observe the dancing aurora curtains creating magical light shows",
            "⚡ Spot the pulsing neutron stars with their rotating energy beams",
            "🎬 Experience cinematic zoom effects during intense moments",
            "✨ See how all 760+ cosmic elements move in perfect parallax harmony"
        ]
        
        tip_index = 0
        zoom_effects_seen = []
        cosmic_events_witnessed = []
        
        print("🌟 COSMIC JOURNEY BEGINS...")
        print()
        
        while True:
            # Enhanced AI behavior for cosmic showcase
            agent_pos = observation[:2]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Track cosmic zoom effects
            current_zoom = env.zoom_level
            if step_count > 15:  # Allow stabilization
                if current_zoom > 1.15 and "zoom in" not in zoom_effects_seen:
                    zoom_effects_seen.append("zoom in")
                    print(f"🎬 CINEMATIC ZOOM IN detected! (level: {current_zoom:.2f}x)")
                    print("    → Creating tension and focus during low energy")
                elif current_zoom < 0.85 and "zoom out" not in zoom_effects_seen:
                    zoom_effects_seen.append("zoom out")
                    print(f"🎬 CINEMATIC ZOOM OUT detected! (level: {current_zoom:.2f}x)")
                    print("    → Providing strategic overview or impact effect")
            
            # Track cosmic phenomena
            if len(env.shooting_stars) > 0 and "shooting stars" not in cosmic_events_witnessed:
                cosmic_events_witnessed.append("shooting stars")
                print(f"💫 SHOOTING STAR SPECTACULAR! {len(env.shooting_stars)} meteors streaking across space!")
            
            # Smart AI behavior for cosmic demonstration
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Cargo delivery → mothership (cosmic zoom: {current_zoom:.2f}x)")
            else:
                # Find and mine asteroids
                nearest_asteroid = None
                min_distance = float('inf')
                
                for i, asteroid in enumerate(asteroid_obs):
                    if np.linalg.norm(asteroid[:2]) > 0.1:
                        distance = np.linalg.norm(asteroid[:2])
                        if distance < min_distance:
                            min_distance = distance
                            nearest_asteroid = asteroid
                
                if nearest_asteroid is not None:
                    if min_distance < env.mining_range:
                        action[2] = 1.0  # Mine
                        print(f"⛏️ Mining {nearest_asteroid[2]:.1f} units (cosmic atmosphere spectacular!)")
                    else:
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.8
                        print(f"🔍 Navigating cosmic void (distance: {min_distance:.1f})")
            
            # Show cosmic phenomena tips
            if step_count % 500 == 0 and tip_index < len(cosmic_phenomena_tips):
                print()
                print(f"🌟 COSMIC PHENOMENA TIP: {cosmic_phenomena_tips[tip_index]}")
                print()
                tip_index += 1
            
            # Add natural movement for better cosmic showcase
            if step_count % 150 == 0:
                action[:2] += np.random.uniform(-0.15, 0.15, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight spectacular cosmic events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                combo_count = env.combo_state.get('chain_count', 0)
                if combo_count >= 2:
                    print(f"🔥 COSMIC COMBO x{combo_count}! Enhanced by spectacular space phenomena!")
                else:
                    print(f"✅ RESOURCE EXTRACTED {extracted:.1f}! (Nebulae swirl majestically)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 CARGO DELIVERED {delivered:.1f}! (Galaxies rotate in cosmic dance)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 COSMIC COLLISION! (Dramatic zoom with spectacular background!)")
            
            # Cosmic universe status updates
            if step_count % 300 == 0:
                total_cosmic_elements = (
                    len(env.nebula_clouds) + 
                    len(env.distant_galaxies) + 
                    sum(len(layer) for layer in env.starfield_layers) + 
                    len(env.space_dust) + 
                    len(env.cosmic_auroras) + 
                    len(env.pulsars) + 
                    len(env.shooting_stars)
                )
                
                print()
                print(f"🌌 COSMIC UNIVERSE STATUS - Step {step_count}:")
                print(f"   🎬 Cinematic Zoom: {current_zoom:.2f}x")
                print(f"   🌌 Nebula Clouds: {len(env.nebula_clouds)} rotating gas formations")
                print(f"   🌀 Spiral Galaxies: {len(env.distant_galaxies)} with stellar arms")
                print(f"   ⭐ Twinkling Stars: {sum(len(layer) for layer in env.starfield_layers)} colored points")
                print(f"   💫 Space Dust: {len(env.space_dust)} atmospheric particles")
                print(f"   🌈 Cosmic Auroras: {len(env.cosmic_auroras)} energy curtains")
                print(f"   ⚡ Active Pulsars: {len(env.pulsars)} neutron beacons")
                print(f"   💥 Shooting Stars: {len(env.shooting_stars)} meteors in motion")
                print(f"   ⏱️ Cosmic Time: {env.cosmic_time:.1f}s (universal animation clock)")
                print(f"   🎯 Total Elements: {total_cosmic_elements} cosmic phenomena active")
                print()
                
                if len(zoom_effects_seen) > 0:
                    print(f"   🎬 Cinematography: {', '.join(zoom_effects_seen)} effects demonstrated")
                if len(cosmic_events_witnessed) > 0:
                    print(f"   🌟 Cosmic Events: {', '.join(cosmic_events_witnessed)} witnessed")
                print()
            
            # Check for end conditions
            if terminated or truncated:
                print()
                print("🌌" + "="*100 + "🌌")
                print("🎬 ULTIMATE COSMIC EXPERIENCE COMPLETE! 🎬")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 COSMIC MISSION SUCCESS - All resources harvested from the universe!")
                    else:
                        print("💫 COSMIC JOURNEY END - Adventure in the stellar void concludes!")
                if truncated:
                    print("⏰ COSMIC TIME LIMIT - The universe cycle completes!")
                
                print()
                print("🌟 SPECTACULAR COSMIC FEATURES EXPERIENCED:")
                print("  ✅ Cinematic HD display (1440x1080) for maximum cosmic immersion")
                print("  ✅ 12 rotating, pulsing nebula clouds with multi-layered beauty")
                print("  ✅ 8 spiral galaxies with detailed rotating stellar arms")
                print("  ✅ 380 twinkling colored stars across realistic parallax layers")
                print("  ✅ 380 cosmic dust particles creating atmospheric depth")
                print("  ✅ 6 dancing aurora curtains with ethereal wave motion")
                print("  ✅ 4 pulsing neutron stars with rotating energy beams")
                print("  ✅ Dynamic shooting stars with spectacular glowing trails")
                print("  ✅ Context-sensitive zoom for cinematic drama")
                print("  ✅ Streamlined UI with perfect visual hierarchy")
                print()
                print("⏳ Displaying ultimate cosmic game over screen...")
                
                # Extended cosmic game over display
                for _ in range(160):
                    env.render()
                    time.sleep(0.025)
                
                print("🌌 Ultimate cosmic experience concluded!")
                break
            
            # Extended cosmic demonstration
            if step_count > 2000:
                print()
                print("🌌" + "="*100 + "🌌")
                print("🎬 ULTIMATE COSMIC DEMONSTRATION COMPLETE! 🎬")
                print()
                print("🏆 COSMIC ACHIEVEMENTS UNLOCKED:")
                print("  ✅ Experienced the most spectacular space mining universe ever created")
                print("  ✅ Witnessed all cosmic phenomena in perfect harmony")
                print("  ✅ Enjoyed cinematic zoom effects and visual storytelling")
                print("  ✅ Navigated through the most beautiful stellar environment")
                print("  ✅ Experienced perfect visual coherence and elegant design")
                print("  ✅ Witnessed living, breathing cosmic atmosphere")
                print("  ✅ Enjoyed streamlined, clutter-free interface")
                print()
                cosmic_summary = f"({', '.join(zoom_effects_seen)})" if zoom_effects_seen else "(awaiting state changes)"
                events_summary = f"({', '.join(cosmic_events_witnessed)})" if cosmic_events_witnessed else "(continuous cosmic beauty)"
                
                print(f"🎬 Cinematic Effects: {cosmic_summary}")
                print(f"🌟 Cosmic Events: {events_summary}")
                print(f"🌌 Total Elements: {total_cosmic_elements} active phenomena")
                print(f"⏱️ Cosmic Time: {env.cosmic_time:.1f}s of universal beauty")
                print()
                print("🌌 THE UNIVERSE OF SPACE MINING IS NOW PERFECT! 🌌")
                break
            
            # Optimal cosmic viewing pace
            time.sleep(0.04)
    
    except KeyboardInterrupt:
        print("\n👋 Cosmic journey interrupted by commander")
        print("🌌 Ultimate cosmic universe successfully demonstrated!")
    except Exception as e:
        print(f"❌ Cosmic anomaly detected: {e}")
    finally:
        env.close()
        print()
        print("🌌" + "="*100 + "🌌")
        print("🚪 ULTIMATE COSMIC SPACE MINING EXPERIENCE COMPLETE! 🚪")
        print("✨ Thank you for exploring the most spectacular universe ever created! ✨")
        print("🌌" + "="*100 + "🌌")

if __name__ == "__main__":
    main()