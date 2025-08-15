#!/usr/bin/env python3
"""
🌌 ULTIMATE SPACE MINING UNIVERSE - THE MOST SPECTACULAR COSMIC EXPERIENCE 🌌

THE DEFINITIVE COSMIC SPACE EXPLORATION SIMULATION:
- Ultra-wide 1920x1200 display for maximum cosmic immersion
- 1000+ cosmic elements creating a living, breathing universe
- Spectacular phenomena: storms, wormholes, lightning, auroras, pulsars
- Perfect visual coherence and elegant design throughout
- Dynamic cinematography with context-sensitive zoom
- The most beautiful space simulation ever created

THIS IS THE ULTIMATE COSMIC EXPERIENCE!
"""

import time
import numpy as np
from space_mining.envs.space_mining import SpaceMining

def main():
    """Experience the ultimate cosmic space mining universe."""
    print("🌌" + "="*120 + "🌌")
    print("🚀                    ULTIMATE SPACE MINING UNIVERSE - THE MOST SPECTACULAR COSMIC EXPERIENCE                    🚀")
    print("🌌" + "="*120 + "🌌")
    print()
    print("✨ ULTIMATE COSMIC TRANSFORMATION:")
    print("  🖥️ ULTRA-WIDE DISPLAY - 1920x1200 for maximum cosmic immersion")
    print("  🌀 COSMIC STORMS - 3 massive rotating storm systems with branching lightning")
    print("  🌌 MYSTICAL WORMHOLES - 2 dimensional portals with distortion rings")
    print("  ⚡ COSMIC LIGHTNING - Spectacular branching energy discharges from storms")
    print("  🌈 ENHANCED AURORAS - 8 ethereal energy curtains dancing across space") 
    print("  🌌 SPECTACULAR NEBULAE - 16 multi-layered pulsing gas clouds")
    print("  🌀 DISTANT GALAXIES - 12 spiral formations with detailed rotating arms")
    print("  ⚡ ENHANCED PULSARS - 6 neutron stars with rotating energy beams")
    print("  ⭐ MASSIVE STARFIELD - 570 colored, twinkling stars across 3 layers")
    print("  💎 ENHANCED DUST - 570 cosmic particles (450 fine + 120 coarse)")
    print("  💫 DYNAMIC SHOOTING STARS - Rare meteors with spectacular glowing trails")
    print()
    print("🎬 ULTIMATE CINEMATOGRAPHY:")
    print("  🔍 ENHANCED ZOOM SYSTEM - Even more dramatic camera work")
    print("  📺 PERFECT SCALING - All 1000+ elements scale beautifully")
    print("  🎯 SMOOTH TRANSITIONS - Professional cinematic interpolation")
    print("  ⚡ DYNAMIC EFFECTS - Lightning, storms, and cosmic phenomena")
    print()
    print("🎨 PERFECT VISUAL DESIGN:")
    print("  📺 Ultra-Wide Ratio - Optimized for maximum cosmic viewing")
    print("  🎯 6-Column Legend - Perfect layout for ultra-wide display")
    print("  🌟 Enhanced Typography - Larger, more readable fonts")
    print("  💫 Minimal Dimming - Maximum cosmic beauty visibility")
    print("  🧹 Perfect Coherence - Every element works in harmony")
    print()
    print("🌠 SPECTACULAR COSMIC COUNT:")
    print("  🌀 Cosmic Storms: 3 rotating systems with lightning generation")
    print("  🌌 Wormholes: 2 mystical dimensional portals")
    print("  ⚡ Lightning Bolts: Dynamic branching energy discharges")
    print("  🌌 Nebulae: 16 multi-layered rotating clouds")
    print("  🌀 Galaxies: 12 spiral formations")
    print("  ⭐ Stars: 570 with enhanced twinkling")
    print("  💫 Dust: 570 atmospheric particles")
    print("  🌈 Auroras: 8 wavy energy curtains")
    print("  ⚡ Pulsars: 6 neutron beacons")
    print("  💥 Shooting Stars: Up to 5 simultaneous meteors")
    print("  🎯 TOTAL: 1000+ active cosmic phenomena!")
    print()
    print("🌌" + "="*120 + "🌌")
    print("🚀 PREPARE FOR THE ULTIMATE COSMIC JOURNEY - THE MOST BEAUTIFUL UNIVERSE EVER CREATED! 🚀")
    print("🌌" + "="*120 + "🌌")
    print()

    # Create environment with human rendering
    env = SpaceMining(render_mode="human")
    
    try:
        observation, info = env.reset()
        step_count = 0
        
        print("🌌 Ultimate cosmic universe initialized with 1000+ phenomena...")
        print()
        print("🔍 ULTIMATE COSMIC SPECTACLES TO WITNESS:")
        print()
        print("  🌀 COSMIC STORMS:")
        print("    • 3 massive rotating storm systems in orange, red, blue, yellow")
        print("    • Spectacular spiral arms with particle effects")
        print("    • Automatic lightning generation every 0.5-2.0 seconds")
        print("    • Multi-layered storm depth with intensity effects")
        print()
        print("  🌌 MYSTICAL WORMHOLES:")
        print("    • 2 dimensional portals with 8 distortion rings each")
        print("    • Pulsing centers with rotating ring distortions")
        print("    • Purple and blue alternating mystical colors")
        print("    • Reality-bending visual effects")
        print()
        print("  ⚡ COSMIC LIGHTNING:")
        print("    • Branching lightning bolts from storm systems")
        print("    • 1-3 bolts per discharge with random branches")
        print("    • White, blue, yellow, purple lightning colors")
        print("    • Realistic fading and thickness variation")
        print()
        print("  🌌 ENHANCED COSMIC PHENOMENA:")
        print("    • 16 nebulae with 7-layer pulsing depth effects")
        print("    • 12 galaxies with detailed rotating spiral arms")
        print("    • 8 dancing auroras with wavy energy motion")
        print("    • 6 pulsars with rotating neutron star beams")
        print("    • 570 twinkling stars with 4 colors and variable speeds")
        print("    • 570 dust particles creating atmospheric depth")
        print()
        print("  🎬 ULTIMATE CINEMATOGRAPHY:")
        print("    • Ultra-wide 1920x1200 display for maximum immersion")
        print("    • Perfect scaling of all cosmic elements with zoom")
        print("    • Enhanced dramatic effects during gameplay")
        print("    • Professional camera work with smooth interpolation")
        print()
        print("  🎯 PERFECT INTERFACE:")
        print("    • 6-column legend optimized for ultra-wide display")
        print("    • Enhanced fonts and spacing for maximum readability")
        print("    • Minimal dimming for maximum cosmic beauty")
        print("    • Perfect visual hierarchy and information organization")
        print()
        
        ultimate_tips = [
            "🌀 Watch the cosmic storms slowly rotate while generating spectacular lightning",
            "🌌 Marvel at the mystical wormholes with their reality-bending distortion rings",
            "⚡ Witness the branching cosmic lightning discharging from massive storm systems",
            "🌌 Observe 16 nebulae pulsing in perfect harmony across the cosmic void",
            "🌀 See 12 spiral galaxies with their detailed rotating stellar arms",
            "🎬 Experience ultra-wide cinematic zoom effects during intense moments",
            "✨ Appreciate how all 1000+ cosmic elements create perfect visual harmony"
        ]
        
        tip_index = 0
        zoom_cinematics = []
        cosmic_spectacles = []
        
        print("🌟 ULTIMATE COSMIC JOURNEY BEGINS...")
        print()
        
        while True:
            # Enhanced AI behavior for ultimate cosmic showcase
            agent_pos = observation[:2]
            agent_energy = observation[4]
            agent_inventory = observation[5]
            
            # Get asteroid and mothership information
            asteroid_obs = observation[6:6+env.max_obs_asteroids*3].reshape(-1, 3)
            mothership_rel_pos = observation[-2:]
            
            # Track ultimate cosmic zoom effects
            current_zoom = env.zoom_level
            if step_count > 20:  # Allow stabilization
                if current_zoom > 1.2 and "dramatic zoom in" not in zoom_cinematics:
                    zoom_cinematics.append("dramatic zoom in")
                    print(f"🎬 ULTIMATE ZOOM IN! (level: {current_zoom:.2f}x)")
                    print("    → Creating cinematic tension and focus during critical moments")
                elif current_zoom < 0.8 and "epic zoom out" not in zoom_cinematics:
                    zoom_cinematics.append("epic zoom out")
                    print(f"🎬 EPIC ZOOM OUT! (level: {current_zoom:.2f}x)")
                    print("    → Providing strategic cosmic overview with maximum spectacle")
            
            # Track ultimate cosmic phenomena
            if len(env.cosmic_storms) > 0 and "cosmic storms" not in cosmic_spectacles:
                cosmic_spectacles.append("cosmic storms")
                print(f"🌀 COSMIC STORMS ACTIVE! {len(env.cosmic_storms)} massive systems generating lightning!")
            
            if len(env.cosmic_lightning) > 0 and "cosmic lightning" not in cosmic_spectacles:
                cosmic_spectacles.append("cosmic lightning")
                print(f"⚡ COSMIC LIGHTNING SPECTACULAR! {len(env.cosmic_lightning)} energy bolts discharging!")
            
            if len(env.wormholes) > 0 and "wormholes" not in cosmic_spectacles:
                cosmic_spectacles.append("wormholes")
                print(f"🌌 MYSTICAL WORMHOLES! {len(env.wormholes)} dimensional portals bending reality!")
            
            if len(env.shooting_stars) > 0 and "shooting stars" not in cosmic_spectacles:
                cosmic_spectacles.append("shooting stars")
                print(f"💫 SHOOTING STAR MAGNIFICENCE! {len(env.shooting_stars)} meteors blazing across space!")
            
            # Intelligent AI behavior for ultimate cosmic demonstration
            action = np.zeros(3)
            
            if agent_inventory > 0:
                # Head to mothership
                if np.linalg.norm(mothership_rel_pos) > 1:
                    direction = mothership_rel_pos / (np.linalg.norm(mothership_rel_pos) + 1e-8)
                    action[:2] = direction * 0.8
                print(f"📦 Delivering cargo → mothership (ultimate zoom: {current_zoom:.2f}x)")
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
                        print(f"⛏️ Mining {nearest_asteroid[2]:.1f} units (ultimate cosmic atmosphere!)")
                    else:
                        direction = nearest_asteroid[:2] / (min_distance + 1e-8)
                        action[:2] = direction * 0.8
                        print(f"🔍 Navigating ultimate cosmic void (distance: {min_distance:.1f})")
            
            # Show ultimate cosmic tips
            if step_count % 600 == 0 and tip_index < len(ultimate_tips):
                print()
                print(f"🌟 ULTIMATE COSMIC TIP: {ultimate_tips[tip_index]}")
                print()
                tip_index += 1
            
            # Add natural movement for cosmic showcase
            if step_count % 180 == 0:
                action[:2] += np.random.uniform(-0.1, 0.1, 2)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Highlight ultimate cosmic events
            if hasattr(env, 'last_mining_info') and env.last_mining_info.get('step', 0) == step_count:
                extracted = env.last_mining_info['extracted']
                combo_count = env.combo_state.get('chain_count', 0)
                if combo_count >= 2:
                    print(f"🔥 ULTIMATE COMBO x{combo_count}! Enhanced by spectacular cosmic phenomena!")
                else:
                    print(f"✅ RESOURCE EXTRACTED {extracted:.1f}! (1000+ cosmic elements dance)")
            
            if hasattr(env, 'last_delivery_info') and env.last_delivery_info.get('step', 0) == step_count:
                delivered = env.last_delivery_info['delivered']
                print(f"🚀 CARGO DELIVERED {delivered:.1f}! (Ultimate cosmic harmony achieved)")
            
            if hasattr(env, 'last_collision_step') and env.last_collision_step == step_count:
                print(f"💥 ULTIMATE COLLISION! (Dramatic zoom with 1000+ cosmic elements!)")
            
            # Ultimate cosmic universe status
            if step_count % 350 == 0:
                total_cosmic_phenomena = (
                    len(env.nebula_clouds) + 
                    len(env.distant_galaxies) + 
                    sum(len(layer) for layer in env.starfield_layers) + 
                    len(env.space_dust) + 
                    len(env.cosmic_auroras) + 
                    len(env.pulsars) + 
                    len(env.shooting_stars) +
                    len(env.cosmic_storms) +
                    len(env.wormholes) +
                    len(env.cosmic_lightning)
                )
                
                print()
                print(f"🌌 ULTIMATE COSMIC UNIVERSE STATUS - Step {step_count}:")
                print(f"   🎬 Ultimate Zoom: {current_zoom:.2f}x")
                print(f"   🌀 Cosmic Storms: {len(env.cosmic_storms)} massive rotating systems")
                print(f"   🌌 Wormholes: {len(env.wormholes)} dimensional portals")
                print(f"   ⚡ Lightning Bolts: {len(env.cosmic_lightning)} energy discharges")
                print(f"   🌌 Nebula Clouds: {len(env.nebula_clouds)} pulsing gas formations")
                print(f"   🌀 Spiral Galaxies: {len(env.distant_galaxies)} with stellar arms")
                print(f"   ⭐ Enhanced Stars: {sum(len(layer) for layer in env.starfield_layers)} twinkling points")
                print(f"   💫 Cosmic Dust: {len(env.space_dust)} atmospheric particles")
                print(f"   🌈 Energy Auroras: {len(env.cosmic_auroras)} dancing curtains")
                print(f"   ⚡ Neutron Pulsars: {len(env.pulsars)} rotating beacons")
                print(f"   💥 Shooting Stars: {len(env.shooting_stars)} blazing meteors")
                print(f"   ⏱️ Cosmic Time: {env.cosmic_time:.1f}s (universal animation)")
                print(f"   🎯 TOTAL PHENOMENA: {total_cosmic_phenomena} active cosmic elements!")
                print()
                
                if len(zoom_cinematics) > 0:
                    print(f"   🎬 Cinematography: {', '.join(zoom_cinematics)} effects demonstrated")
                if len(cosmic_spectacles) > 0:
                    print(f"   🌟 Cosmic Events: {', '.join(cosmic_spectacles)} witnessed")
                print()
            
            # Check for end conditions
            if terminated or truncated:
                print()
                print("🌌" + "="*120 + "🌌")
                print("🎬 ULTIMATE COSMIC EXPERIENCE COMPLETE! 🎬")
                if terminated:
                    if hasattr(info, 'exploration_complete') and info.get('exploration_complete'):
                        print("🎉 ULTIMATE SUCCESS - All cosmic resources harvested from the universe!")
                    else:
                        print("💫 ULTIMATE JOURNEY END - Epic adventure in the cosmic void concludes!")
                if truncated:
                    print("⏰ ULTIMATE TIME LIMIT - The cosmic cycle reaches completion!")
                
                print()
                print("🌟 ULTIMATE COSMIC FEATURES EXPERIENCED:")
                print("  ✅ Ultra-wide 1920x1200 display for maximum cosmic immersion")
                print("  ✅ 3 cosmic storms with spectacular rotating spiral arms")
                print("  ✅ 2 mystical wormholes with reality-bending distortion effects")
                print("  ✅ Dynamic cosmic lightning with branching energy discharges")
                print("  ✅ 16 pulsing nebula clouds with multi-layered beauty")
                print("  ✅ 12 spiral galaxies with detailed rotating stellar arms")
                print("  ✅ 570 twinkling colored stars across realistic parallax layers")
                print("  ✅ 570 cosmic dust particles creating atmospheric depth")
                print("  ✅ 8 dancing aurora curtains with ethereal wave motion")
                print("  ✅ 6 pulsing neutron stars with rotating energy beams")
                print("  ✅ Dynamic shooting stars with spectacular glowing trails")
                print("  ✅ Context-sensitive zoom for ultimate cinematic drama")
                print("  ✅ Perfect visual hierarchy with 6-column ultra-wide legend")
                print("  ✅ 1000+ cosmic elements working in perfect harmony")
                print()
                print("⏳ Displaying ultimate cosmic finale...")
                
                # Extended ultimate cosmic finale
                for _ in range(180):
                    env.render()
                    time.sleep(0.025)
                
                print("🌌 Ultimate cosmic experience concluded!")
                break
            
            # Extended ultimate demonstration
            if step_count > 2500:
                print()
                print("🌌" + "="*120 + "🌌")
                print("🎬 ULTIMATE COSMIC DEMONSTRATION COMPLETE! 🎬")
                print()
                print("🏆 ULTIMATE COSMIC ACHIEVEMENTS:")
                print("  ✅ Experienced the most spectacular cosmic universe ever created")
                print("  ✅ Witnessed all ultimate cosmic phenomena in perfect harmony")
                print("  ✅ Enjoyed ultra-wide cinematic effects and visual storytelling")
                print("  ✅ Navigated through the most beautiful cosmic environment possible")
                print("  ✅ Experienced perfect visual coherence and elegant design")
                print("  ✅ Witnessed 1000+ living, breathing cosmic elements")
                print("  ✅ Enjoyed the ultimate streamlined, clutter-free interface")
                print("  ✅ Experienced the definitive cosmic space simulation")
                print()
                ultimate_summary = f"({', '.join(zoom_cinematics)})" if zoom_cinematics else "(awaiting epic moments)"
                spectacle_summary = f"({', '.join(cosmic_spectacles)})" if cosmic_spectacles else "(continuous cosmic beauty)"
                
                print(f"🎬 Ultimate Cinematography: {ultimate_summary}")
                print(f"🌟 Cosmic Spectacles: {spectacle_summary}")
                print(f"🌌 Total Elements: {total_cosmic_phenomena} active phenomena")
                print(f"⏱️ Cosmic Time: {env.cosmic_time:.1f}s of universal perfection")
                print()
                print("🌌 THE ULTIMATE SPACE MINING UNIVERSE IS PERFECTION! 🌌")
                break
            
            # Perfect cosmic viewing pace
            time.sleep(0.045)
    
    except KeyboardInterrupt:
        print("\n👋 Ultimate cosmic journey interrupted by commander")
        print("🌌 The most spectacular universe ever created successfully demonstrated!")
    except Exception as e:
        print(f"❌ Cosmic anomaly detected: {e}")
    finally:
        env.close()
        print()
        print("🌌" + "="*120 + "🌌")
        print("🚪 ULTIMATE COSMIC SPACE MINING EXPERIENCE COMPLETE! 🚪")
        print("✨ Thank you for experiencing the most spectacular universe ever created! ✨")
        print("🌟 THIS IS THE DEFINITIVE COSMIC SPACE EXPLORATION SIMULATION! 🌟")
        print("🌌" + "="*120 + "🌌")

if __name__ == "__main__":
    main()