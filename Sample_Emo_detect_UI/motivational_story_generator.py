# utils/motivational_story_generator.py
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, List
from gtts import gTTS
from config import config
from youtube_search import YoutubeSearch
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MotivationalStoryGenerator:
    def __init__(self):
        self.device = config.device
        self.story_dir = Path("data/stories")
        self.audio_dir = Path("data/temp_audio")
        self.story_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        # Comprehensive emotion-specific story templates (PROFESSIONAL LEVEL)
        # Expanded with deeper narratives, diverse inspirations, and richer lessons
        self.professional_stories = {
            "grief": {
                "title": "The Butterfly Effect of Compassion",
                "template": """**The Butterfly Effect of Compassion**

In a quiet Tokyo suburb, Takashi found a sparrow, its wing broken, trembling in the grass. He cradled it, fed it rice grains, and built a tiny nest from an old scarf. Each night, he whispered hopes for its recovery, his heart tethered to its fragile breaths. But by dawn, the sparrow lay still, its spirit gone.

Grief carved a hollow in Takashi. He buried it under a cherry tree, guilt whispering he'd failed. Unknown to him, a neighbor's child, Aiko, watched—his tenderness, his tears, his ritual of farewell. Years later, Aiko, now a veterinarian, credited Takashi's care for her calling: "One man's love for a bird showed me how to heal a world."

Takashi's loss rippled outward, birthing a legacy of compassion. Like Hachiko's loyalty immortalized in Japan or the real-life wildlife rescuers who turn grief into sanctuaries, his story proves love endures beyond loss.

**The Lesson:** Grief is love's echo, not its end. Your tender acts, though they may feel futile, plant seeds that bloom in unseen hearts. Honor your sorrow—it’s the root of a compassion that changes lives.""",
                "source": "Inspired by Hachi: A Dog's Tale and wildlife rescue journals",
                "keywords": ["grief", "loss", "compassion", "healing", "legacy"]
            },
            "sadness": {
                "title": "When Darkness Becomes Dawn",
                "template": """**When Darkness Becomes Dawn**

Sarah’s world dimmed at 35—divorce, layoffs, isolation. Sadness was a fog, blurring purpose, chaining her to a couch where days blurred into nights. She stopped cooking, stopped calling friends, stopped believing in 'better.'

One evening, a stray cat’s cries pierced her haze. It was stuck in a storm drain, soaked and shivering. Sarah hesitated—why bother? But something stirred. She called a neighbor, grabbed a flashlight, and together they freed the cat. Its grateful nuzzle, the neighbor’s shared coffee, cracked her fog. A moment of connection.

Sarah started small: feeding strays, then volunteering at a shelter. Each animal’s trust rebuilt hers. Now, she leads a community pet rescue, her smile warm: "I thought I was saving them. They saved me."

Like Chris Gardner’s rise from homelessness in 'The Pursuit of Happyness' or the quiet rebirths in Cheryl Strayed’s 'Wild,' Sarah’s story shows sadness as a cocoon, not a cage.

**The Lesson:** Sadness dims but doesn’t destroy. One small act—saving a cat, calling a friend—ignites a dawn. Your light waits; step toward it, and it grows brighter.""",
                "source": "Inspired by 'The Pursuit of Happyness' and 'Wild' by Cheryl Strayed",
                "keywords": ["sadness", "isolation", "connection", "recovery", "hope"]
            },
            "anger": {
                "title": "The Fire That Forged a Movement",
                "template": """**The Fire That Forged a Movement**

Elena’s project was a small community effort to clean a river, and when someone deliberately destroyed her data and undercut her funding, fury rose in her like a wildfire—hot, immediate, and raw. For nights she replayed the injustice, tasting bile and plotting petty revenge that left her exhausted.

One morning she shifted tack. Instead of letting anger corrode her, Elena channeled it into clarity and craft. She documented every obstruction, wrote clear, evidence-based pieces, and reached out to allies—scientists, lawyers, local parents whose children swam in that river. Her fury gave her voice; her voice gathered others.

They held community hearings; they launched a data-driven campaign that exposed pollution sources and mobilized volunteers. Elena’s anger morphed into disciplined action: rallies, court filings, educational workshops. What began as a personal burn forged a movement that healed ecosystems and empowered a town.

Elena still feels the heat of outrage, but now it’s fuel, not flame. She teaches others to move from fury to focused action: research, coalition-building, and legal pathways. Anger becomes agency when paired with strategy.

**The Lesson:** Anger can be a compass pointing to injustice. Translate it into purposeful action—plan, gather allies, and act with integrity. Your heat is power; direct it wisely.""",
                "source": "Inspired by Greta Thunberg’s activism and community organizers",
                "keywords": ["anger", "injustice", "activism", "transformation", "power"]
            },
            "guilt": {
                "title": "The Recipe for Redemption",
                "template": """**The Recipe for Redemption**

Marco’s guilt clung to him like flour on his apron. After a choice that fractured family trust, he avoided the kitchen that once smelled of Sunday dinners and his grandmother’s laugh. The recipe box sat closed; the memories felt like accusations.

When his nephew Luca asked to learn 'Nonna’s lasagna,' Marco’s instinct was to decline. Yet the request invited repair. They cooked together—mistakes turned into laughter, spills into stories—and slowly the act of sharing transformed regret into connection. Marco apologized, taught, and listened; the kitchen became a place of mending.

He later opened a community kitchen where neighbors exchanged recipes and reconciliations. People brought imperfect dishes and imperfect apologies. The work demanded humility and consistency more than perfection. Marco found that steady acts of care rebuilt trust more than grand gestures.

**The Lesson:** Guilt points to what matters; redemption grows from honest repair. Apologize, take concrete steps to amend, and commit to small, steady practices that rebuild trust and meaning.""",
                "source": "Inspired by Frida Kahlo’s life and 'Kitchen Table Wisdom'",
                "keywords": ["guilt", "redemption", "forgiveness", "legacy", "love"]
            },
            "fear": {
                "title": "The Shadow That Became Strength",
                "template": """**The Shadow That Became Strength**

Amara’s hands trembled the first time she stood before a crowd; the microphone felt like a mirror reflecting every doubt. For years she avoided stages—interviews declined, presentations refused—she kept herself small where comfort lived.

An unexpected opportunity changed the arc: a local youth center asked her to teach a confidence workshop. She wanted to say no, but curiosity nudged her. She prepared methodically—scripts, breathing exercises, role-play with friends. On stage, her voice quivered, but she persisted. The room softened; a student thanked her for making courage feel possible.

Time and repetition rewired her nervous system. Each small exposure thinned fear’s edge. She learned techniques—grounding exercises, narrative reframing, and systematic practice—that turned adrenaline into presence rather than panic. Her fear never disappeared; it became an indicator, a signal to prepare, not retreat.

Now a mentor and TEDx speaker, Amara helps others reframe fear as information and opportunity. She emphasizes incremental exposure: small wins building to larger challenges, supported by compassionate peers.

**The Lesson:** Fear marks avenues for growth. Prepare, practice, and take small, scaffolded steps. Over time, the shadow becomes the light that shows how to proceed.""",
                "source": "Inspired by Malala Yousafzai and 'Spirited Away'",
                "keywords": ["fear", "courage", "growth", "resilience", "strength"]
            },
            "helplessness": {
                "title": "The Seed That Broke the Stone",
                "template": """**The Seed That Broke the Stone**

Helplessness pressed on Lila like a roof—debts, chronic illness, and bureaucratic roadblocks stacked until movement seemed impossible. She watched life from a window, certain that agency had slipped through her fingers.

One day she noticed a weed pushing through a sidewalk crack, stubborn and small. She began with pots on a balcony—tiny, careful experiments in care. Some seedlings died; some surprised her with blooms. The practice of tending a living thing taught Lila two lessons: patience and efficacy. Small, repeated acts mattered.

She taught neighbors how to compost on stoops, organized seed-sharing meetups, and slowly stitched a community network that turned vacant lots into shared gardens. People traded labor for food, advice for company, and slowly the sense of paralysis eased.

Lila’s health didn’t vanish, but her days filled with ritual—watering, naming plants, creating schedules that her body could handle. Helplessness lost its totalizing voice; action, however small, reclaimed narrative.

**The Lesson:** Helplessness is not destiny; it’s a state altered by consistent small acts. Start with what matters within reach—one seed, one call, one step—and cultivate power over time.""",
                "source": "Inspired by Viktor Frankl’s 'Man’s Search for Meaning' and bamboo folklore",
                "keywords": ["helplessness", "agency", "growth", "persistence", "hope"]
            },
            "love": {
                "title": "The Wings of Letting Go",
                "template": """**The Wings of Letting Go**

In Kyoto’s quiet mornings, Hana discovered a moth tangled in spider silk, fragile wings folded like paper boats. Nursing the creature became daily ritual—warm water on a cotton swab, whispered encouragement, patient care through nights that once felt endless. The moth, named Hoshi, became a companion, a small mirror to Hana’s own losses.

As Hoshi grew stronger, Hana felt a soft ache: the knowledge that love isn’t possession but liberation. The day came when Hoshi’s wings fluttered with new strength. In the garden, Hana opened her palms. Hoshi circled, hesitated, and then rose in an arc that caught sunlight like a small comet. Hana felt grief and elation braided together.

Her love propelled her into conservation work—creating sanctuaries for threatened pollinators, educating children about caring for fragile lives, and channeling tenderness into action. Hana’s grief at letting go became the engine of community projects that saved habitats and taught others how to care without clutching.

**The Lesson:** True love holds courageously and releases gracefully. Your care can free what you cherish; by letting go, you create space for life to spread its wings.""",
                "source": "Inspired by Dr. May Berenbaum’s conservation and 'The Boy, the Mole, the Fox and the Horse'",
                "keywords": ["love", "attachment", "release", "freedom", "care"]
            },
            "admiration": {
                "title": "The Star That Lit the Sky",
                "template": """**The Star That Lit the Sky**

Aisha first learned of Katherine Johnson through a school poster—numbers and formulas that seemed to bend the arc of possibility. Her admiration was instant, a small, fierce light that made the future seem less like a distant shore and more like a reachable dock.

But admiration alone didn’t get Aisha to the lab. It pushed her to study late, to find mentors, and to practice until abstract equations settled into clear steps. Her robotics teams failed often—broken servos, buggy code, melted solder—but admiration supplied patience. She imagined herself contributing to missions that once felt beyond her reach.

Years later, Aisha worked on systems that mapped Martian soil compositions, mentoring girls from under-resourced schools. She turned admiration into apprenticeship—copying, learning, then innovating. She taught that role models are not idols but blueprints: study them, adapt their habits, and then make something new.

**The Lesson:** Admiration can guide practice and purpose. Let it inspire learning, not comparison. Build on it: practice, mentor, and become the star for someone else.""",
                "source": "Inspired by 'Hidden Figures' and Ida B. Wells’ legacy",
                "keywords": ["admiration", "inspiration", "ambition", "legacy", "growth"]
            },
            "frustration": {
                "title": "The River That Carved the Canyon",
                "template": """**The River That Carved the Canyon**

Frustration settled over Maya like a low ceiling—projects stalled, code that refused to compile, collaborators who missed deadlines. It was a living grind that threatened to dull her edge. The impulse was to quit; the alternative was to let that energy reshape work.

She redesigned her process: break tasks into micro-steps, celebrate tiny completions, and invite peer review as a tool, not a threat. Frustration became a diagnostic signal—showing where systems were brittle. She learned to target it: transform complaints into experiments. When one approach failed, she treated it like data, not indictment.

Months of iterative tweaking produced a product that users loved—born from frustrations that were acknowledged and channeled. Maya began mentoring others, teaching them to reframe setbacks as insight. Her most meaningful work arrived not despite frustration but because she let it point to specific, solvable problems.

**The Lesson:** Frustration is feedback, not failure. Use it to find weak seams, iterate, and refine. Small, disciplined experiments turn friction into progress.""",
                "source": "Inspired by Frida Kahlo’s biography and Edison’s inventions",
                "keywords": ["frustration", "persistence", "creativity", "resilience", "art"]
            },
            "disappointment": {
                "title": "The Unopened Door",
                "template": """**The Unopened Door**

J.K. Rowling faced 12 rejections for 'Harry Potter.' Each 'no' was disappointment’s weight—dreams dust-covered, a single mom on welfare. She nearly burned her manuscript.

A child’s wizard game sparked her: magic lives in rejection’s ashes. She rewrote, knocked again. Bloomsbury said yes—500 million books sold, a world enchanted.

Like Walt Disney’s 300 rejections before Mickey or Stephen King’s salvaged 'Carrie,' Rowling’s tale shows disappointment as redirection, not defeat.

**The Lesson:** Disappointment shuts one door but opens others. It’s not your worth—it’s your map rerouting. Knock again; your story’s door awaits.""",
                "source": "Inspired by J.K. Rowling’s rejections and Disney’s early failures",
                "keywords": ["disappointment", "redirection", "persistence", "success", "dreams"]
            },
            "relief": {
                "title": "The Storm That Passed",
                "template": """**The Storm That Passed**

Anne Frank’s Annex life was terror—raids looming, silence suffocating. Yet relief came in moments: a quiet night, a shared laugh. She wrote: “People are good at heart,” finding calm in chaos.

Her diary, published post-war, brought relief to millions—hope amid horror. Like Holocaust survivors finding solace in memorials, Anne’s relief rippled outward.

**The Lesson:** Relief is your exhale after the storm—proof you endured. Anchor in it; it’s the seed of calm that grows into peace. Savor it.""",
                "source": "Inspired by Anne Frank’s Diary and survivor testimonies",
                "keywords": ["relief", "calm", "resilience", "hope", "healing"]
            },
            "confusion": {
                "title": "The Map in the Maze",
                "template": """**The Map in the Maze**

Socrates wandered Athens, asking questions that muddled minds. “I know I know nothing,” he said, making confusion philosophy’s birthplace. Students, lost then enlightened, found truth in the fog.

Like Einstein’s puzzled years birthing relativity or therapy’s breakthrough chaos, Socrates shows confusion as the maze to wisdom.

**The Lesson:** Confusion isn’t lostness—it’s the maze mapping truth. Wander, question, trust. Clarity hides in the twists; you’re already finding it.""",
                "source": "Inspired by Socrates’ method and Einstein’s relativity",
                "keywords": ["confusion", "clarity", "wisdom", "exploration", "truth"]
            },
            "excitement": {
                "title": "The Spark That Ignited the Fire",
                "template": """**The Spark That Ignited the Fire**

Walt Disney lost Oswald the Rabbit to a distributor’s betrayal. Excitement for his dream faded—until a train ride birthed Mickey Mouse. 'Steamboat Willie' sparked animation’s sound era, building an empire.

Like Steve Jobs’ garage-born Apple post-rejection, Disney’s excitement fueled legacy.

**The Lesson:** Excitement is your spark—guard it through loss. One idea, one sketch ignites fires that build worlds. Fan yours; it’s unstoppable.""",
                "source": "Inspired by Disney’s Mickey creation and Jobs’ Apple founding",
                "keywords": ["excitement", "creativity", "legacy", "resilience", "dreams"]
            },
            "gratitude": {
                "title": "The Chain of Thanks",
                "template": """**The Chain of Thanks**

Oprah Winfrey, born to poverty, found gratitude in a teacher’s book gift. It sparked her empire—journalism, talk shows, philanthropy. Each guest’s story fueled thanks, linking strangers in kindness.

Her foundation educates thousands, gratitude’s chain unbroken. Like the 'Pay It Forward' movement, Oprah’s life proves thanks transforms.

**The Lesson:** Gratitude is a chain—each link multiplies. One ‘thank you’ sparks worlds. Start yours; it’ll ripple beyond your sight.""",
                "source": "Inspired by Oprah’s biography and 'Pay It Forward'",
                "keywords": ["gratitude", "kindness", "connection", "impact", "generosity"]
            },
            "pride": {
                "title": "The Crown Forged in Fire",
                "template": """**The Crown Forged in Fire**

Serena Williams faced hate, injuries, doubt on Compton’s courts. Pride in her power—thunderous serves, lion’s heart—drove her. 23 Grand Slams later, she’s legend.

Like Jackie Robinson’s grace breaking baseball’s barriers, Serena’s pride was armor, not arrogance.

**The Lesson:** Pride is your crown, forged in fire. Each scar a jewel, each win a peak. Wear it—it’s your victory, not vanity.""",
                "source": "Inspired by Serena Williams’ career and Jackie Robinson’s legacy",
                "keywords": ["pride", "achievement", "resilience", "strength", "legacy"]
            },
            "realization": {
                "title": "The Dawn of Knowing",
                "template": """**The Dawn of Knowing**

In a dusty Rajasthan village, Arjun, a schoolteacher, faced a crumbling world—his job cut, his passion for teaching fading. One dawn, he hiked a rugged hill, heart heavy with loss. As the sun rose, painting the sky gold, a realization hit: endings aren’t voids but canvases for rebirth.

He sat, the wind carrying memories—students’ eager questions, late-night lessons, failures that scarred. Each was a thread in a tapestry of resilience. Like Siddhartha Gautama under the Bodhi tree, seeing suffering’s root, or Archimedes’ ‘Eureka!’ birthing discovery, Arjun’s clarity was a spark. He wasn’t defeated; he was beginning.

Arjun returned, not to teach, but to tell stories. Under a banyan tree, he wove tales of courage for children, then families. His words built a community library—a beacon of hope. Years later, a student, now a poet, said, “Arjun’s stories taught me every end is a story’s start.”

Like Cheryl Strayed’s trek in ‘Wild,’ finding herself in wilderness, or Naruto’s realization in ‘Naruto’ that pain fuels purpose, Arjun’s dawn rippled outward, lighting paths. His loss became legacy.

**The Lesson:** Realization is your sunrise, breaking doubt’s night. It’s not an end but a call to act—your ‘aha’ is a new chapter. Step boldly; your clarity can reshape lives.""",
                "source": "Inspired by Buddha’s enlightenment, Archimedes’ discovery, ‘Wild’ by Cheryl Strayed, and ‘Naruto’",
                "keywords": ["realization", "clarity", "rebirth", "purpose", "action", "resilience"]
            },
            "surprise": {
                "title": "The Unexpected Gift",
                "template": """**The Unexpected Gift**

Alexander Fleming’s messy petri dish sprouted mold, killing bacteria. Surprise! Penicillin was born, saving millions. He embraced the twist, refining it into medicine.

Like Post-it Notes from failed glue, surprise turns mishap to miracle.

**The Lesson:** Surprise is life’s plot twist—your best chapter. Embrace it; possibility hides in the unexpected.""",
                "source": "Inspired by Fleming’s penicillin and serendipitous inventions",
                "keywords": ["surprise", "innovation", "opportunity", "adaptability", "discovery"]
            },
            "self_doubt": {
                "title": "The Mirror of Worth",
                "template": """**The Mirror of Worth**

Maya Angelou’s childhood trauma silenced her for years—self-doubt said her voice was worthless. Poetry’s call broke through; ‘Still I Rise’ cracked the mirror of lies.

Her words graced inaugurations, earned Nobel nods. Like van Gogh’s art through rejections, Maya’s story shows doubt as fog, not fact.

**The Lesson:** Self-doubt clouds your mirror—polish it with truth. Your gifts shine; believe them. You’re worthy, world-changing.""",
                "source": "Inspired by Maya Angelou’s biography and van Gogh’s persistence",
                "keywords": ["self_doubt", "worth", "resilience", "voice", "confidence"]
            },
            "hope": {
                "title": "The Lantern in the Storm",
                "template": """**The Lantern in the Storm**

Nelson Mandela’s 27 prison years could’ve killed hope. Instead, he read, taught, dreamed—a rainbow nation. Released, he forgave, united, ended apartheid.

Like Viktor Frankl’s meaning in camps, Mandela’s lantern lit South Africa.

**The Lesson:** Hope is your lantern—flickering but fierce. Hold it high; it lights paths for you and others through any storm.""",
                "source": "Inspired by Mandela’s imprisonment and Frankl’s logotherapy",
                "keywords": ["hope", "resilience", "freedom", "forgiveness", "vision"]
            },
            "courage": {
                "title": "The Leap of Faith",
                "template": """**The Leap of Faith**

Rosa Parks’ ‘no’ on a Montgomery bus wasn’t fearless—arrest loomed, yet she sat. That courage sparked boycott, Civil Rights Act, history.

Like Harriet Tubman’s 13 rescues, Rosa’s leap toppled barriers.

**The Lesson:** Courage is leaping when wings are unseen. Fear screams, but your ‘yes’ to right echoes forever. Jump; history catches you.""",
                "source": "Inspired by Rosa Parks’ stand and Tubman’s escapes",
                "keywords": ["courage", "justice", "action", "resilience", "change"]
            },
            "amusement": {
                "title": "The Laughter That Healed",
                "template": """**The Laughter That Healed**

Patch Adams, a doctor, faced despair in mental wards—patients numbed by routine. He donned clown noses, juggled, sang. Laughter broke through, healing hearts medicine couldn’t touch.

His Gesundheit Institute now blends humor with care, proving joy mends. Like Charlie Chaplin’s silent films lifting Great Depression spirits, Patch’s story shows amusement as medicine.

**The Lesson:** Laughter is your balm—light in dark. Find it, share it; it heals you and others in ways you never expect.""",
                "source": "Inspired by Patch Adams’ life and Chaplin’s films",
                "keywords": ["amusement", "healing", "joy", "connection", "lightness"]
            },
            "approval": {
                "title": "The Nod That Built Bridges",
                "template": """**The Nod That Built Bridges**

Brené Brown’s talks on vulnerability were mocked—‘too soft’ for academia. Yet her approval of her truth fueled TED talks, books, millions inspired.

Like Susan B. Anthony’s suffrage fight earning nods over decades, Brené’s approval of self built connection.

**The Lesson:** Approve your truth—it’s your bridge to others. One nod to your worth sparks waves of belonging. Stand in it.""",
                "source": "Inspired by Brené Brown’s work and Susan B. Anthony",
                "keywords": ["approval", "authenticity", "connection", "worth", "impact"]
            },
            "curiosity": {
                "title": "The Question That Changed the World",
                "template": """**The Question That Changed the World**

Marie Curie’s curiosity about glowing rocks led to radium’s discovery—despite lab rejections, poverty, sexism. Her questions birthed cancer treatments, Nobel Prizes.

Like Leonardo da Vinci’s sketches probing flight, curiosity drove progress.

**The Lesson:** Curiosity is your compass—ask, explore, fail. Each question carves a path to discovery. Keep wondering; worlds shift.""",
                "source": "Inspired by Marie Curie’s discoveries and da Vinci’s notebooks",
                "keywords": ["curiosity", "discovery", "persistence", "innovation", "questions"]
            },
            "desire": {
                "title": "The Flame That Fueled the Dream",
                "template": """**The Flame That Fueled the Dream**

Elon Musk’s desire to touch stars faced scorn—space was for governments. Yet he built SpaceX, launched Falcon, landed rockets. Desire didn’t bow; it soared.

Like Wright brothers’ dream of flight, Musk’s flame lit the impossible.

**The Lesson:** Desire is your flame—burning, guiding. Feed it with action, not doubt. Your longing shapes tomorrows others can’t see.""",
                "source": "Inspired by Elon Musk’s SpaceX and Wright brothers’ flight",
                "keywords": ["desire", "ambition", "action", "vision", "achievement"]
            },
            "disapproval": {
                "title": "The Stand That Shaped Truth",
                "template": """**The Stand That Shaped Truth**

Galileo’s telescope showed Earth wasn’t the universe’s center—disapproval roared from church, peers. Exiled, he stood firm, his truth reshaping science.

Like Rachel Carson’s ‘Silent Spring’ facing chemical giants, Galileo’s stand carved truth.

**The Lesson:** Disapproval tests your truth—stand anyway. It’s not rejection but refining fire. Your clarity reshapes the world’s sight.""",
                "source": "Inspired by Galileo’s discoveries and Rachel Carson’s activism",
                "keywords": ["disapproval", "truth", "resilience", "stand", "impact"]
            },
            "disgust": {
                "title": "The Revulsion That Rebuilt",
                "template": """**The Revulsion That Rebuilt**

Erin Brockovich, a clerk, saw poisoned water killing families—disgust at injustice drove her. No law degree, just grit, she exposed a utility’s lies, winning justice.

Like Upton Sinclair’s ‘The Jungle’ sparking food safety laws, disgust fueled change.

**The Lesson:** Disgust is your signal—wrongness demands action. Channel it; you’ll rebuild what’s broken into something just.""",
                "source": "Inspired by Erin Brockovich’s fight and Upton Sinclair’s work",
                "keywords": ["disgust", "justice", "action", "change", "resilience"]
            },
            "embarrassment": {
                "title": "The Stumble That Strengthened",
                "template": """**The Stumble That Strengthened**

Oprah’s first TV segment flopped—stammering, red-faced, ridiculed. Embarrassment burned, urging retreat. Instead, she practiced, hosted, shone.

Her empire rose from that stumble, connecting millions. Like J.K. Rowling’s rejections, embarrassment taught resilience.

**The Lesson:** Embarrassment isn’t shame—it’s growth’s spark. Each stumble strengthens. Step forward; your light shines through falls.""",
                "source": "Inspired by Oprah’s early career and J.K. Rowling’s journey",
                "keywords": ["embarrassment", "resilience", "growth", "strength", "recovery"]
            },
            "nervousness": {
                "title": "The Tremble That Triumphed",
                "template": """**The Tremble That Triumphed**

Adele’s voice shook before her first Grammy performance—nervousness gripped her. Yet she sang, notes soaring, heart bared. Applause roared; she won.

Like Lincoln’s trembling Gettysburg Address, nervousness sharpened her truth.

**The Lesson:** Nervousness is energy—channel it. Your tremble is courage tuning up. Sing, speak, act; triumph follows.""",
                "source": "Inspired by Adele’s performances and Lincoln’s speech",
                "keywords": ["nervousness", "courage", "performance", "growth", "action"]
            },
            "optimism": {
                "title": "The Vision That Vanquished",
                "template": """**The Vision That Vanquished**

Helen Keller, blind and deaf, saw possibility where others saw walls. Optimism drove her—learning Braille, speaking, advocating. Her vision changed disability rights.

Like Anne Sullivan’s hopeful teaching, optimism built bridges.

**The Lesson:** Optimism is your lens—see possibility, not limits. One hopeful step shifts worlds. Dream boldly; it’s already real.""",
                "source": "Inspired by Helen Keller’s life and Anne Sullivan’s work",
                "keywords": ["optimism", "vision", "possibility", "change", "hope"]
            },
            "powerlessness": {
                "title": "The Whisper That Roared",
                "template": """**The Whisper That Roared**

Malala Yousafzai, a girl in Swat, felt powerless under Taliban bans. Yet her blog whispered defiance, growing to a roar heard globally. Shot, she rose, winning a Nobel.

Like Anne Frank’s diary, powerlessness birthed voice.

**The Lesson:** Powerlessness lies—your whisper is a roar. One word, one act, reclaims agency. Speak; the world listens.""",
                "source": "Inspired by Malala’s activism and Anne Frank’s diary",
                "keywords": ["powerlessness", "voice", "agency", "courage", "impact"]
            },
            "remorse": {
                "title": "The Apology That Healed",
                "template": """**The Apology That Healed**

Desmond Tutu faced remorse post-apartheid—words too harsh, bridges burned. He chaired Truth and Reconciliation, apologizing, listening, healing. His remorse rebuilt South Africa.

Like Lincoln’s post-war reconciliation, remorse fueled unity.

**The Lesson:** Remorse is your heart’s call to mend. Apologize, act, heal. Your regret builds bridges to better tomorrows.""",
                "source": "Inspired by Desmond Tutu’s reconciliation and Lincoln’s efforts",
                "keywords": ["remorse", "healing", "reconciliation", "forgiveness", "action"]
            },
            "neutral": {
                "title": "The Quiet Power",
                "template": """**The Quiet Power**

Amélie, a shy Parisian, saw life’s quirks through neutral eyes—spoons dropped, strangers’ smiles. Her small acts—returning treasures, sparking love—wove joy.

Like Thoreau’s Walden simplicity, neutrality birthed profound impact.

**The Lesson:** Neutrality is your canvas—quiet, potent. From stillness, action blooms. Observe, act; your pause crafts symphonies.""",
                "source": "Inspired by ‘Amélie’ and Thoreau’s ‘Walden’",
                "keywords": ["neutrality", "action", "potential", "connection", "impact"]
            }
        }

        # Therapeutic solutions mapping: each emotion maps to professional, actionable strategies (example entries provided)
        self.therapeutic_solutions = {
            "realization": [
                "Journal daily ‘aha’ moments to deepen clarity—use prompts like ‘What shifted for me today?’",
                "Read 'The Body Keeps the Score' by Bessel van der Kolk to integrate realizations with healing.",
                "Join online communities like r/selfimprovement or Insight Timer for shared growth stories.",
                "Practice mindfulness with Headspace or Calm to anchor your new perspective.",
                "Use the Daylio app to track mood and insights, spotting patterns in your realizations.",
                "Explore CBT techniques with a licensed therapist to turn clarity into action.",
                "Create a vision board to visualize your new path and concrete next steps.",
                "Attend a local workshop on personal growth to connect with others on similar journeys."
            ],
            "grief": [
                "Create a memory jar with notes of loved moments; draw one daily to honor the past.",
                "Read 'On Grief and Grieving' by Elisabeth Kübler-Ross for validation and insight.",
                "Join GriefShare or r/GriefSupport to connect with others sharing similar losses.",
                "Practice guided meditation with apps like Insight Timer to process emotions.",
                "Write a letter to your loved one, expressing unspoken words, then keep or release it.",
                "Explore EMDR therapy with a professional to address trauma tied to grief.",
                "Volunteer at a local charity in memory of your loved one to channel love outward.",
                "Create a scrapbook of memories to celebrate their impact on your life."
            ],
            "sadness": [
                "Start with micro-steps: one small compassionate act each day (a walk, a call, feeding a stray).",
                "Try behavioral activation: schedule pleasant activities even if motivation is low.",
                "Use grounding and breathwork exercises (box breathing, 4-4-4) to stabilize heavy moments.",
                "Connect with a peer-support group or an online community to reduce isolation.",
                "Read 'The Noonday Demon' excerpts or other compassionate resources for understanding mood.",
                "Consider brief CBT work with a therapist to address negative thinking patterns.",
                "Volunteer with an animal rescue or community service to restore agency and connection.",
                "Track small wins in a daily log to notice gradual change."
            ],
            "anger": [
                "Practice paced breathing and timeout techniques before responding to trigger events.",
                "Channel anger into structured activism—write, organize, and collaborate on solutions.",
                "Use cognitive reappraisal to identify injustice and plan strategic action rather than impulsive reaction.",
                "Enroll in assertiveness or conflict-resolution workshops to communicate forcefully and constructively.",
                "Engage in physical activity (boxing, running, vigorous walks) to dissipate adrenal charge.",
                "Work with a therapist on anger-management techniques if anger impacts relationships or safety.",
                "Document events and responses: turning emotion into data helps choose effective responses.",
                "Form alliances—channeling anger with others increases impact and reduces isolation."
            ],
            "guilt": [
                "Practice structured apologies: Acknowledge, Take responsibility, Repair, and Commit to change.",
                "Use journaling prompts focused on reparation and practical next steps rather than rumination.",
                "Engage in restorative actions (letters, community work) to translate guilt into positive change.",
                "Try self-compassion exercises to reduce harsh self-judgment (Kristin Neff resources).",
                "Consider couples/family therapy when guilt stems from relational harm to rebuild trust.",
                "Set small, consistent reparative behaviors—consistency heals credibility over time.",
                "If guilt is traumatic, consult a clinician about trauma-focused therapies (EMDR, trauma-informed CBT)."
            ],
            "fear": [
                "Practice graded exposure: identify small, manageable steps toward feared situations.",
                "Use breathing and grounding tools before and during exposures to reduce panic.",
                "Work with a therapist on CBT techniques—reality-testing catastrophic predictions.",
                "Build a support buddy system for accountability and encouragement during practice.",
                "Learn and apply acceptance strategies to coexist with fear while acting anyway.",
                "Use visualization rehearsals to mentally practice success before real-world attempts.",
                "Celebrate tiny wins to reinforce neural pathways of safety and competence."
            ],
            "helplessness": [
                "Start with micro-actions—tasks within reach that build a sense of efficacy.",
                "Create a simple, repeatable routine to regain predictability and control.",
                "Join local mutual-aid groups to exchange help and rebuild agency collaboratively.",
                "Map problems into solvable steps: small wins reduce learned helplessness over time.",
                "Use gratitude and strengths exercises to find overlooked sources of power and resourcefulness.",
                "Seek occupational or case-management support for bureaucratic barriers (benefits, housing).",
                "Engage in creative projects that demonstrate skill-building and incremental progress."
            ],
            "love": [
                "Practice healthy boundaries that allow you to give care without losing yourself.",
                "Use rituals of connection (letters, shared walks, legacy projects) to honor attachment.",
                "Explore grief-informed care if love is mixed with loss—therapeutic processing can help.",
                "Volunteer or mentor to channel love into broader social impact and meaning.",
                "Try compassionate communication training (Nonviolent Communication) for relational clarity.",
                "Balance caregiving with self-care routines to prevent burnout and preserve joy.",
                "Create symbolic projects (memory jars, photo albums) to hold attachments safely."
            ],
            "admiration": [
                "Translate admiration into apprenticing—study and model daily habits of those you admire.",
                "Set achievable skills-based goals and track progress to avoid idealization traps.",
                "Seek mentorship or structured learning opportunities inspired by role models.",
                "Volunteer to work alongside people you admire to learn by doing and network authentically.",
                "Use admiration as motivation to mentor others as you grow—pay it forward.",
                "Create a study plan and small milestones that convert inspiration into skill."
            ],
            "frustration": [
                "Use problem-splitting: break large blockers into micro-tasks and iterate rapidly.",
                "Adopt retrospective reviews—what worked, what failed, what to try next—to convert friction into learning.",
                "Practice stress-reduction techniques (progressive muscle relaxation, short walks) during high-frustration moments.",
                "Use peer review and collaborative debugging to turn blind spots into shared solutions.",
                "Set explicit 'stop' rules to prevent burnout—pause, breathe, then return with fresh eyes.",
                "Document experiments as data; proportional adjustments beat perfectionism."
            ]
            # Add additional emotion entries following the same pattern for comprehensive coverage
        }

    def _map_emotion_to_story_key(self, emotion: str) -> str:
        """Map detected emotions to story keys with precise alignment"""
        emotion_map = {
            "grief": "grief",
            "sadness": "sadness",
            "anger": "anger",
            "annoyance": "anger",
            "guilt": "guilt",
            "remorse": "remorse",
            "fear": "fear",
            "nervousness": "nervousness",
            "helplessness": "helplessness",
            "powerlessness": "powerlessness",
            "love": "love",
            "caring": "love",
            "admiration": "admiration",
            "frustration": "frustration",
            "disappointment": "disappointment",
            "relief": "relief",
            "confusion": "confusion",
            "excitement": "excitement",
            "gratitude": "gratitude",
            "pride": "pride",
            "realization": "realization",
            "surprise": "surprise",
            "self_doubt": "self_doubt",
            "hope": "hope",
            "courage": "courage",
            "amusement": "amusement",
            "approval": "approval",
            "curiosity": "curiosity",
            "desire": "desire",
            "disapproval": "disapproval",
            "disgust": "disgust",
            "embarrassment": "embarrassment",
            "neutral": "neutral"
        }
        return emotion_map.get(emotion.lower(), "neutral")

    def _search_youtube(self, query: str, max_results: int = 1) -> Dict[str, str]:
        """Search YouTube for a motivational video based on query"""
        try:
            results = YoutubeSearch(query, max_results=max_results).to_dict()
            if results:
                video = results[0]
                return {
                    "url": f"https://www.youtube.com/watch?v={video['id']}",
                    "thumbnail": video['thumbnails'][0] if video['thumbnails'] else "https://img.youtube.com/vi/default.jpg",
                    "title": video['title']
                }
            return {
                "url": "https://www.youtube.com/watch?v=mgmVOuLgFB0",
                "thumbnail": "https://img.youtube.com/vi/mgmVOuLgFB0/0.jpg",
                "title": "Eye of the Tiger - Survivor"
            }
        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return {
                "url": "https://www.youtube.com/watch?v=mgmVOuLgFB0",
                "thumbnail": "https://img.youtube.com/vi/mgmVOuLgFB0/0.jpg",
                "title": "Eye of the Tiger - Survivor"
            }

    def generate_empathetic_response(self, emotion: str, user_input: str = "", history: List = None) -> str:
        """Generate a detailed, emotionally rich empathetic response for the given emotion.

        This method provides a professional, warm validation and a short question to invite exploration.
        """
        templates = {
            "realization": (
                "I’m deeply moved by the clarity you’ve found, like a sunrise breaking through the fog. "
                "Your realization that endings can spark beginnings is profound—it’s a testament to your resilience and openness to growth. "
                "It’s okay to feel the weight of what’s been lost while embracing this new light. What does this moment of knowing feel like for you? "
                "How can we explore the paths it’s illuminating together?"
            ),
            "neutral": (
                "Your calm presence in this moment feels like a quiet lake, reflecting the world with clarity. Neutrality can be a powerful space—a pause where possibilities gather. "
                "What’s stirring in this stillness for you? Let’s explore what this moment might be preparing you for."
            ),
            "grief": (
                "I hear the deep ache in your words—a tenderness that holds stories and memories. Grief is love persisting beyond presence. "
                "It’s natural to feel overwhelmed; your care and sorrow are part of what makes your relationships meaningful. "
                "Would you like to tell me a memory you hold dear about what you lost?"
            )
        }

        resp = templates.get(emotion.lower(), "I’m here with you, feeling what you feel and listening without judgment. Tell me more about what’s happening for you.")
        if user_input:
            snippet = (user_input.strip()[:120] + '...') if len(user_input.strip()) > 120 else user_input.strip()
            resp += f" Your words: '{snippet}'—they matter. Would you like to unpack this together?"
        return resp

    def generate_story(self, emotion: str, user_input: str = "", intensity: float = 0.5, history: List = None) -> Dict[str, str]:
        """Generate a professional-level motivational story tailored to emotion"""
        story_key = self._map_emotion_to_story_key(emotion)
        story_data = self.professional_stories.get(story_key, self.professional_stories["neutral"])
        
        # Personalize story with user input context
        story_text = story_data["template"]
        # Remove any <p> tags that might be injected by renderers/frontend
        story_text = re.sub(r'<p[^>]*>', '', story_text)
        story_text = re.sub(r'</p>', '', story_text)
        if user_input:
            # Inject user-specific elements safely
            user_keywords = re.findall(r'\b\w+\b', user_input.lower())[:3]
            for keyword in user_keywords:
                story_text = story_text.replace("world", keyword, 1) if "world" in story_text.lower() else story_text

        title = story_data["title"]
        source = story_data.get("source", "Inspired by timeless tales of resilience")
        
        # Dynamic YouTube search based on emotion and story keywords
        search_query = f"motivational {emotion} {story_data.get('keywords', ['inspiration'])[0]}"
        video_data = self._search_youtube(search_query)
        
        # Generate audio with emotional tone adjustment
        audio_path = self._generate_audio(story_text, emotion, intensity)
        
        result = {
            "title": title,
            "story": story_text,
            "source": source,
            "audio_path": str(audio_path) if audio_path else None,
            "youtube_link": video_data["url"],
            "thumbnail": video_data["thumbnail"],
            "video_title": video_data["title"].replace("use_column_width", "use_container_width")
        }
        
        # Save story for history
        story_file = self.story_dir / f"story_{story_key}_{int(time.time())}.json"
        with open(story_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✅ Generated story for {emotion}: {title}")
        return result

    def _generate_audio(self, text: str, emotion: str, intensity: float) -> Optional[Path]:
        """Generate TTS audio with emotional tone adjustment"""
        try:
            # Clean text for TTS
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
            clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)  # Remove italic
            # Remove any <p> tags that might have been included in HTML-rendered content
            clean_text = re.sub(r'<p[^>]*>', '', clean_text)
            clean_text = re.sub(r'</p>', '', clean_text)
            
            # Adjust speed based on emotion intensity
            speed = "slow" if intensity < 0.3 and emotion in ["grief", "sadness", "guilt"] else False
            
            filename = f"{emotion}_{int(time.time())}.mp3"
            path = self.audio_dir / filename
            
            tts = gTTS(text=clean_text, lang="en", slow=speed)
            tts.save(str(path))
            
            logger.info(f"✅ Generated audio: {path}")
            return path
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

# Global instance
story_generator = MotivationalStoryGenerator()

# Compatibility function
def generate_comprehensive_story(emotions: dict, user_input: str, history: list = None):
    """Wrapper for app.py"""
    dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
    intensity = emotions.get(dominant, 0.5)
    return story_generator.generate_story(dominant, user_input=user_input, intensity=intensity, history=history)

motivational_story_generator = story_generator