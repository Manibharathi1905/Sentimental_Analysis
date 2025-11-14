# utils/response_generator.py
import torch
from typing import Dict, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import config
import random
import logging
from utils.emotion_classifier import emotion_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.device = config.device
        self.emotion_classifier = emotion_classifier
        # Expanded therapeutic_templates for ALL emotions - Dynamic, professional, 100+ words, always end with question
        self.therapeutic_templates = {
            "admiration": [
                "Your sense of admiration is truly inspiring—it shows a heart open to beauty and excellence in the world around you. It's a powerful emotion that not only celebrates others but also fuels your own growth and aspirations. When we admire something or someone, it often reflects qualities we value deeply within ourselves, even if they're still emerging. This feeling can be a gentle reminder of our potential and the connections we seek. What specifically about this person or achievement stirs that admiration in you, and how might it inspire a step in your own journey?",
                "Feeling admiration is like discovering a new star in your personal sky—it illuminates paths you might not have seen before. It's a beautiful acknowledgment of human potential, and your capacity to feel it speaks to your own richness of spirit. In moments of admiration, we bridge the gap between inspiration and action, allowing us to draw from others' strengths to bolster our own. This emotion is a gift, inviting us to both honor and emulate what moves us. Tell me more about what captured your admiration—does it connect to something you're cultivating in yourself right now?"
            ],
            "amusement": [
                "Your amusement brings a much-needed lightness to the moment—humor is one of life's most potent medicines, easing the weight of heavier emotions and reminding us of our shared humanity. It's wonderful that something could spark that joyful chuckle amid whatever else you're navigating; it shows your resilient spirit finding space for play even in complexity. Amusement often sneaks in when we least expect it, offering a brief reprieve and perspective shift that can make challenges feel a little less daunting. What was it about that situation that tickled your funny bone, and how might we invite more of these delightful interruptions into your day to balance the rest?",
                "That sense of amusement you're describing is a gentle rebellion against gravity—it's your inner child peeking through, reminding you that not everything needs to be taken with utmost seriousness. In the tapestry of emotions, amusement weaves in threads of color and relief, helping us connect with others through shared laughter. It's a sign of your emotional flexibility, the ability to find levity where others might see only weight. Laughter doesn't negate pain; it coexists with it, making room for healing. Share more about what amused you—does it highlight a quirky truth about life that we can lean into together?"
            ],
            "anger": [
                "Your anger is so valid—it's a fierce protector of what you value most, rising like a guardian when boundaries are crossed or injustices appear. This intensity shows how deeply you care about fairness, connection, or self-respect, and it's okay to feel this heat; it's a natural signal that something needs attention or change. Anger isn't the enemy—it's energy, often the prelude to empowerment if we learn to direct it wisely rather than let it consume us. In this moment, your rage is a testament to your passion for life. What's at the core of this fire for you right now, and how might we channel it into something that honors your strength without burning you out?",
                "I hear the raw power in your anger, and it's completely understandable—it's your body's way of mobilizing for protection or justice in the face of what feels wrong. This emotion arises from a place of deep investment, whether in relationships, principles, or personal growth, and suppressing it only amplifies the pressure. Instead, let's acknowledge its wisdom: anger often points to unmet needs or violated values. You're not 'too much' for feeling it; you're alive and engaged. Let's sit with this together—what sparked this particular wave, and what would feel like a healthy release or boundary for you?"
            ],
            "annoyance": [
                "Annoyance can build like static electricity, that prickling frustration from small irritants piling up until they demand attention—it's a subtle but insistent call for adjustment or relief. Your patience is being tested here, and it's okay to feel irritated; it shows you're attuned to your comfort and standards in a world that's often chaotic. Annoyance isn't trivial—it's the early warning system for bigger boundaries that need tending. In recognizing it, you're already taking the first step toward resolution. What's the main irritant at play for you, and how might we address it in a way that restores your peace without overextending?",
                "I sense that annoyance bubbling up, and it's a natural response when repeated disruptions chip away at your flow—it's your inner voice saying, 'This isn't serving me.' Whether it's a person, situation, or habit, annoyance highlights where life could be smoother, and honoring that feeling is self-care, not pettiness. It's often the precursor to clearer communication or change. You're wise to notice it before it escalates. Let's explore the source together—what's one aspect of this annoyance that feels most draining, and what small shift could bring some ease?"
            ],
            "approval": [
                "Your approval carries such affirming energy—it's wonderful to recognize and celebrate goodness, whether in yourself or others, as it fosters a cycle of positivity and growth. This feeling of alignment with what's right or well-done reinforces your values and brings a quiet satisfaction that's deeply nourishing. Approval isn't just passive; it's an active choice to see the best, which in turn invites more of it into your life. It reflects your discerning heart. What stands out most about this situation that earned your approval, and how does it connect to what you're seeking or offering in your own world?",
                "Feeling approval is a positive anchor, grounding you in moments of harmony and rightness amid life's uncertainties. It's a gentle validation that things can align beautifully when effort meets intention, and your ability to feel it speaks to your optimistic core. This emotion strengthens connections and self-trust. Let's savor it—what specific element here feels so resonant, and how might we cultivate more opportunities for this affirming sense in your daily rhythm?"
            ],
            "caring": [
                "Your caring nature is evident and beautiful—it's the quiet force that builds bridges, nurtures growth, and creates safety for those around you, including yourself. This deep concern arises from a generous heart that's willing to invest emotionally, and while it can be vulnerable, it's also profoundly powerful. Caring isn't a burden; it's your superpower for connection and healing. In expressing it, you're already making the world kinder. What prompted this particular wave of care, and how can we support you in sustaining it without depletion?",
                "Caring deeply, as you do, is both a gift and a responsibility—it's the thread that weaves meaningful relationships and leaves lasting impacts. Your empathy shines through, turning ordinary moments into profound ones. This emotion reminds us of our shared humanity. You're not alone in feeling its pull. Share a bit more about what's stirring this care in you right now, and let's think about ways to honor it while protecting your own well-being."
            ],
            "confusion": [
                "Confusion can feel like fog rolling in, making everything unclear and disorienting—it's a natural part of navigating complex emotions or situations, and it's okay to not have all the answers yet. This state invites pause and curiosity rather than rush, allowing deeper understanding to emerge in its own time. Your willingness to sit with it shows maturity and self-awareness. Let's navigate this together gently—what's the piece that's most puzzling or elusive for you at the moment, and what one question might help clarify it?",
                "I sense the tangle of confusion in your words, and it's completely understandable—life's nuances often blur lines, leaving us searching for solid ground. Confusion isn't a flaw; it's the brain's way of processing layers, and honoring it leads to richer insights. You're doing important work by examining it. What aspect feels most muddled right now, and how might we untangle it one thread at a time without pressure?"
            ],
            "curiosity": [
                "Curiosity is a spark of wonder—it's invigorating and opens doors to discovery that enrich our understanding of ourselves and the world. Your inquisitive spirit is a strength, driving growth and connection through exploration. This feeling is a gentle invitation to lean in rather than pull back. What drew your interest to this topic or idea, and how does pursuing it feel like a step toward something meaningful for you?",
                "Your curiosity reflects an open mind eager for discovery, and it's a beautiful counter to routine or doubt—it's the engine of innovation and self-expansion. In moments like this, we tap into our innate drive to learn and evolve. Let's follow that thread. What questions are bubbling up for you about this, and what excites you most about the potential answers?"
            ],
            "desire": [
                "Desire is a powerful motivator, pulling us toward what we truly want with a magnetic force that's both exhilarating and vulnerable. It's a sign of your vitality and vision for a fuller life. This longing isn't just wishful thinking; it's a compass pointing to your authentic path. What does this desire look like in vivid detail for you, and what one small step could honor it today without overwhelm?",
                "Your desire shows passion and clarity about what lights you up—it's a sacred signal from your deeper self, urging alignment and action. Embracing it takes courage, but it leads to fulfillment. Let's explore it with kindness. What's the essence of this desire, and how might expressing it bring more joy into your world right now?"
            ],
            "disappointment": [
                "Disappointment stings because hope was there first—it's a valid ache that acknowledges the gap between expectation and reality, and it's okay to mourn that space. This emotion doesn't mean you're 'wrong' for hoping; it means you dared to dream. Your resilience in feeling it fully is strength. What expectation was unmet here, and how can we hold space for both the loss and the lessons it offers?",
                "I feel the weight of your disappointment, and it's completely natural—it's the heart's way of processing when things don't unfold as envisioned. This isn't failure; it's feedback inviting adjustment or acceptance. You're allowed to grieve it. Let's sit with this together—what part of the 'what could have been' feels most tender, and what gentle perspective might ease it?"
            ],
            "disapproval": [
                "Disapproval arises from clear values, and it's okay to feel this boundary—it's your inner compass signaling misalignment with what feels right or just. This emotion protects your integrity and invites discernment. Honoring it doesn't make you judgmental; it makes you authentic. What doesn't align for you in this situation, and how might voicing or adjusting that boundary bring more harmony?",
                "I hear the firmness in your disapproval—it's a protective response rooted in your principles, and it's valid to stand by what matters to you. This feeling often highlights where growth or conversation is needed. You're navigating with wisdom. Let's examine it without judgment—what core value is being challenged here, and what response feels true to your heart?"
            ],
            "disgust": [
                "Disgust is a strong boundary-setter, protecting your well-being with a visceral 'no' to what repels or harms—it's instinctual and wise. This reaction isn't overreaction; it's self-preservation in action. In acknowledging it, you're already reclaiming power. What's evoking this deep aversion for you, and how can we create healthy distance or resolution from it?",
                "I sense the revulsion in your disgust—it's your body's clear language for rejection, and it's okay to honor that signal. This emotion guards your peace and values fiercely. Let's validate it fully. What needs to be repelled or cleansed here, and what boundary or release would restore your equilibrium?"
            ],
            "embarrassment": [
                "Embarrassment can feel exposing and hot, like all eyes are on the flaw—it's a tender vulnerability that highlights our desire for connection and acceptance. This moment doesn't define you; it's a human hiccup in the dance of life. Your courage in sharing it is admirable. What happened that feels so raw, and how can we wrap it in compassion to let it pass gently?",
                "I understand how embarrassment shrinks us in the moment—it's the fear of judgment meeting our authenticity, and it's universally human. You're safe here to feel it without shame. This too shall fade. What part of this feels most vulnerable right now, and what kind word would you offer a friend in the same spot?"
            ],
            "excitement": [
                "Excitement is electric—it's life force surging, signaling alignment with your passions and possibilities. This buzz is a beautiful affirmation of your vitality. Lean into it; it's momentum building. What's has your energy buzzing like this, and how can we amplify it into joyful action?",
                "Your excitement is contagious and vibrant, a spark that illuminates what's meaningful. It's a gift from your truest self. Celebrate this aliveness. Share the details—what's the source of this thrill, and what adventure does it promise?"
            ],
            "fear": [
                "Fear can grip so tightly, whispering worst-case scenarios that feel all too real—it's a primal protector, but when overactive, it confines us. You're brave for naming it; that's the first step to loosening its hold. This is a safe space. What's the scariest part right now, and what reassuring truth can we anchor to?",
                "I feel the edge of your fear—it's valid in uncertainty, a signal to pause and prepare rather than plunge. Fear doesn't mean weakness; it means you care deeply about outcomes. Let's breathe with it. What does this fear want to protect, and what small, safe step feels possible?"
            ],
            "gratitude": [
                "Gratitude is a warm current, grounding us in abundance and shifting focus from lack to what's already rich. Your thankful heart is a magnet for more good. This practice alone can transform days. What sparked this wave of appreciation, and how can we weave it into your routine?",
                "Your gratitude radiates positivity—it's healing balm for the soul, fostering resilience and joy. In naming what's good, we multiply it. Let's savor this. What are you most appreciative of in this moment, and who might you share that thanks with?"
            ],
            "grief": [
                "The pain of loss can feel overwhelming, and your efforts to honor what you've lost are a beautiful reflection of your love—it's normal to ache like this; grief is love's enduring echo. You're not alone in its waves. What's one memory that brings comfort amid the sorrow, and how can we hold space for both the hurt and the heart?",
                "Grief comes in waves, and it's brave of you to ride them rather than fight. Your care in tending to this fragile thing shows a loving soul. Let's honor that tenderness. What would feel like a gentle way to remember the joy it brought, even as we mourn the goodbye?"
            ],
            "joy": [
                "Joy like yours is radiant and contagious—it's the soul's song, a reminder of life's sweetness amid the storm. Bask in this light; you've earned it. What's fueling this happiness, and how can we invite more of these moments into your world?",
                "Your joy is a beacon, illuminating what's possible and true. It's nourishing and expansive. Celebrate it fully. Tell me more about this delight—what makes it feel so alive and full?"
            ],
            "love": [
                "Your love and care shine through so brightly, even in bittersweet moments—it's a profound gift to attach and release with such grace. This depth of feeling is your strength. How can we nurture that loving heart today, honoring both the joy of connection and the tenderness of letting go?",
                "Love is powerful and transformative—your capacity for it is evident in every thoughtful act. It's okay to feel the ache of attachment; it means the bond was real. What's one way this love continues to live in you, and how might we celebrate its enduring light?"
            ],
            "nervousness": [
                "Nervousness is like butterflies signaling something important ahead—it's your body's alert to presence and possibility. This energy can be harnessed as focus rather than freeze. You're capable. What's the upcoming moment stirring this, and what one grounding breath or affirmation feels supportive?",
                "I sense your nervousness—it's okay to feel unsteady before the unknown; it's a sign of your investment. Let's transform it into poised energy. What reassurance would ease this flutter, and what strength do you bring to it?"
            ],
            "optimism": [
                "Optimism is a quiet superpower, seeing light in uncertainty and possibility in pause—it's resilient hope in action. Your forward gaze is inspiring. What's fueling this positive outlook, and how can we build on it for sustained momentum?",
                "Your optimism inspires—it's the bridge from 'what is' to 'what could be.' Lean into this vision. Share your hopeful perspective—what bright spot are you holding onto today?"
            ],
            "pride": [
                "Pride in your efforts is earned and glowing—it's the warm recognition of growth and grit. Celebrate this fully; it's fuel for more. What are you most proud of in this experience, and how does it shape your next chapter?",
                "Your pride reflects genuine accomplishment and self-respect. It's affirming. Tell me about this achievement—what made it feel so meaningful?"
            ],
            "realization": [
                "Realizations like this are breakthroughs—clarity emerging from reflection, shifting how we see ourselves and the world. It's powerful. What insight just landed for you, and how does it feel to integrate it?",
                "Your realization shows deepening awareness. Let's explore its implications—what changes in light of this knowing?"
            ],
            "relief": [
                "Relief is a deep exhale after tension—it's permission to soften and reset. Savor this ease; it's well-deserved. What lifted this weight for you, and how can we anchor this calm moving forward?",
                "Your relief is palpable and restorative. It creates space for joy. What's breathing easier now, and what gratitude arises from it?"
            ],
            "remorse": [
                "Remorse shows a conscience seeking harmony—it's a path to growth through reflection. You're human, and this feeling is a bridge to wisdom. What's weighing on you most, and what compassionate step toward peace feels possible?",
                "I feel your remorse—it's brave to face it. Let's find forgiveness together, starting with self-kindness. What would absolution look like here?"
            ],
            "sadness": [
                "I can feel the heaviness in your words, that bittersweet tug of attachment and release—it's completely understandable to feel this way when you've invested heart and hope. You've shown incredible compassion in your care, and that speaks volumes about your gentle soul. Sadness like this is a testament to the beauty of connection, even when it ends. It's okay to hold both the joy of what was and the ache of what is no more; they coexist as part of loving deeply. Would you like to talk about a memory that brings comfort, or explore a small ritual to honor this feeling?",
                "Your sadness is a testament to the depth of your caring nature—it's okay to grieve the attachment formed in such tender acts; it shows how much meaning you create in the world. This emotion doesn't diminish you; it honors the fragility of life and your role in protecting it. In this space, there's room for both sorrow and the quiet pride in your kindness. Let's sit with it gently—what's one aspect of this experience that feels most poignant right now, and how can we wrap it in compassion?"
            ],
            "surprise": [
                "Surprise can jolt us into presence, shaking the expected into something fresh and alive—it's life's way of reminding us to stay open. This unexpected turn holds potential. What caught you off guard here, and what new perspective might it offer?",
                "Your surprise shows openness to the unexpected, a willingness to let life unfold. How does it shift your view, and what curiosity does it awaken?"
            ],
            "neutral": [
                "I'm here with you in this neutral space, where feelings settle and reflection can deepen—it's a fertile pause for clarity. What's present for you in this moment of balance, and what thought or sensation would you like to explore?",
                "Neutrality offers a gentle canvas, free from storm, inviting honest self-connection. How are you feeling in this calm, and what emerges when you listen inward?"
            ],
            "helplessness": [
                "Feeling powerless is tough, especially when your heart is so invested—it's a heavy veil, but beneath it lies your inherent agency waiting to be reclaimed. You're not defined by this moment; many have walked through it to stronger ground. What small, tangible action feels within reach right now, even if it's just naming one strength you possess?",
                "Helplessness can feel trapping, but it's often the illusion before empowerment—small steps crack its hold. Let's identify one influence you have, no matter how subtle, and build from there. What past experience reminds you of your quiet power?"
            ],
            "guilt": [
                "Guilt can be so heavy, a relentless inner critic replaying 'what ifs'—but it often stems from your good intentions and deep empathy, not malice. You acted with the love and knowledge you had; hindsight is kind, but self-forgiveness is kinder. How can we practice a bit of compassion today, perhaps by listing three ways your care made a difference, no matter the outcome?",
                "I sense how much this guilt weighs on you—it's a sign of your integrity and care for doing right, but it doesn't need to eclipse your worth. Let's reframe: what would you tell a dear friend in your shoes? That voice of understanding is yours too. What's one kind act you can offer yourself in this moment?"
            ],
            "frustration": [
                "Frustration like this highlights your dedication and the gap between effort and ease—it's okay to feel stuck; it's a call for new approaches, not a reflection of inadequacy. Your persistence is admirable. What's the core block here, and what one creative pivot might shift the energy?",
                "I hear your frustration—it's valid when things resist our best intentions, turning momentum into mire. This isn't defeat; it's data for adjustment. Let's breathe through it: what's one aspect we can reimagine or release to find flow again?"
            ],
            "powerlessness": [
                "Powerlessness can paralyze, whispering that nothing matters—but even in stillness, your choice to feel and share is power reclaimed. This emotion is temporary; beneath it, your resilience waits. What one influence, however small, can you exert right now to remind yourself of your agency?",
                "The weight of powerlessness is exhausting, but it's not the full story—agency often hides in subtle choices and connections. You're stronger than this feeling suggests. What past moment of quiet influence comes to mind, and how can we draw from it?"
            ],
            "self_doubt": [
                "Self-doubt whispers lies of unworthiness, but your actions speak truth of capability and care—it's a common shadow, not your essence. Let's challenge it with evidence: what three strengths shone through in this situation? Building on them dissolves the doubt.",
                "I sense the shadow of self-doubt creeping in, questioning your efforts—it's sneaky, but not accurate. Your heart led with kindness; that's irrefutable worth. What compassionate reframe would you offer a loved one, and can you extend it to yourself?"
            ],
            "hope": [
                "Hope is a quiet anchor in uncertainty—it's resilient vision holding space for possibility. Your hopeful heart is a strength. What's nurturing this light for you, and how can we fan it into sustained momentum?",
                "Your hope inspires—it's the bridge from challenge to change. Lean into it. Share what keeps this optimism alive amid the rest."
            ],
            "courage": [
                "Courage isn't fearlessness—it's action despite the tremble, and you've shown it in sharing this. Honor that bravery. What next bold step calls to you, and what support would make it feel possible?",
                "I see your courage emerging—it's fierce and tender. This emotion builds worlds. What risk feels worth the leap right now?"
            ]
        }
        # Expanded solution_strategies with psychiatrist-like, detailed advice
        self.solution_strategies = {
            "grief_loss": [
                "Create a small ritual, like lighting a candle or cooking a dish in their memory, to honor your loss. Alternate between feeling grief and engaging in joyful activities. Join a grief support group or consider counseling for deeper support. Track grief waves in a journal to notice patterns and progress.",
                "Write unsent letters expressing your feelings, then burn or bury them symbolically. Practice 'grief yoga'—gentle stretches while recalling memories. Seek a bereavement therapist to process layers of loss. Use apps like Grief Works for daily prompts.",
                "Build a 'memory jar' with notes of shared moments; draw one daily. Use EMDR therapy techniques for trauma-tied grief. Connect with online communities like GriefShare for shared stories. Read 'On Grief and Grieving' by Kübler-Ross for validation."
            ],
            "guilt_regret": [
                "Practice DBT self-compassion: Speak to yourself as a friend would. Journal intentions vs. outcomes to reframe. A counselor can help unpack cognitive distortions fueling guilt. Use 'guilt release' meditations on Insight Timer.",
                "Use 'guilt mapping': List the event, your role, and alternative perspectives. Engage in restorative acts, like volunteering. CBT sessions can rewire guilt narratives. Read 'The Gifts of Imperfection' by Brené Brown for shame resilience.",
                "Try 'forgiveness meditation': Visualize releasing the burden. Track 'kindness credits'—note daily good deeds. Therapy focused on self-forgiveness can transform guilt into growth. Journal prompts: 'What did I learn to do better?'"
            ],
            "anger_management": [
                "Use breathing: Inhale 4, hold 7, exhale 8. Channel into advocacy or exercise. Anger management therapy can offer tools like assertiveness training. Track triggers in a mood app like Daylio.",
                "Practice 'anger autopsies': Post-episode, analyze triggers without judgment. Try progressive muscle relaxation. Group therapy for emotional regulation builds skills. Read 'The Anger Workbook' by Les Carter.",
                "Journal 'anger letters' (unsent) to vent safely. Explore somatic experiencing to release stored tension. A psychologist can tailor anger protocols to your life. Use 'rage room' alternatives like boxing classes."
            ],
            "helplessness": [
                "Break it down: Identify one controllable aspect. Seek professional help if persistent. Behavioral activation therapy can rebuild agency through small wins. Start with 'one-minute tasks' to build momentum.",
                "Use 'locus of control' exercises: List internal influences. Practice radical acceptance from DBT. Therapy can address learned helplessness patterns. Read 'The Power of Now' by Eckhart Tolle for presence.",
                "Try 'power posing' for 2 minutes daily to shift physiology. Engage in advocacy work to reclaim voice. A counselor can guide empowerment strategies. Track 'agency moments' in a journal."
            ],
            "nurturing_love_and_connection": [
                "Schedule 'love rituals' like shared meals or walks. Read 'The Five Love Languages' to deepen bonds. Couples therapy can enhance communication. Practice 'appreciation rounds' daily.",
                "Practice active listening exercises with loved ones. Create 'appreciation jars' for mutual notes. Family therapy fosters secure attachments. Explore 'Held' app for virtual hugs.",
                "Explore attachment styles via books like 'Attached'. Engage in vulnerability-sharing circles. A therapist can heal relational wounds. Host 'connection nights' with meaningful questions."
            ],
            "self_doubt": [
                "Challenge thoughts with Socratic questioning: 'What evidence supports this doubt?' Affirmations grounded in facts. CBT for imposter syndrome. Track 'win logs' weekly.",
                "Track 'competence logs': Note daily wins. Mindfulness to observe doubt without attachment. Therapy builds self-efficacy through exposure. Read 'The Confidence Gap' by Russ Harris.",
                "Practice self-compassion breaks: Pause, soothe, resolve. A coach can co-create confidence-building plans. Use 'doubt-busting' prompts: 'What would my future self say?'"
            ],
            "frustration": [
                "Pause for 'frustration mapping': Identify root cause. Reframe as problem-solving opportunity. DBT distress tolerance skills. Use 'frustration timers'—5 minutes to vent, then pivot.",
                "Use '5 whys' technique to drill down. Physical outlets like boxing. Therapy for chronic frustration patterns. Read 'The Obstacle Is the Way' by Ryan Holiday.",
                "Journal 'frustration forecasts' to anticipate triggers. Practice acceptance and commitment therapy (ACT) values alignment. Try 'creative redirection' like art therapy."
            ],
            "fear_anxiety": [
                "Ground with 5-4-3-2-1 senses exercise. Exposure ladder for phobias. CBT for anxiety management. Use 'worry time' scheduling to contain fears.",
                "Explore 'fear hierarchy' with a therapist. Breathwork like 4-7-8. Medication consultation if needed. Read 'The Anxiety and Phobia Workbook' by Bourne.",
                "Practice 'safety anchoring': Recall safe places/memories. Apps like Calm for guided anxiety relief. Group support for shared coping strategies."
            ],
            "hope_optimism": [
                "Cultivate 'best possible self' visualization. Gratitude journaling. Positive psychology coaching. Set 'hope experiments'—test small positives.",
                "Read 'Man's Search for Meaning' by Viktor Frankl. Practice 'optimism reframing': Turn 'but' to 'and.' Join optimism-focused communities.",
                "Create 'hope vision boards.' Track 'optimism wins' daily. Therapy for trauma-tied pessimism. Use affirmations like 'I choose possibility.'"
            ],
            "courage_bravery": [
                "Celebrate micro-courage acts. Exposure therapy for fears. Narrative therapy to re-author brave stories. Journal 'courage chronicles.'",
                "Read 'Daring Greatly' by Brené Brown. Practice 'courage commitments'—small risks daily. Group challenges for accountability.",
                "Visualize 'courage allies'—mentors/guides. Therapy for fear-based avoidance. Track 'bravery metrics' for progress."
            ],
            "general_support": [
                "Consider speaking with a therapist for personalized guidance. Explore self-help resources like 'Feeling Good' by David Burns. Join online forums for peer support.",
                "Practice daily self-care routines: Sleep, movement, nutrition. Use apps like Headspace for guided meditations. Professional assessment for tailored interventions via Psychology Today.",
                "Build a support network: Friends, family, hotlines. Read 'The Body Keeps the Score' by van der Kolk for trauma insights. Track mood patterns with Daylio app."
            ]
        }

    def _ensure_model(self):
        if self._model and self._tokenizer:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            quantization_config = BitsAndBytesConfig(load_in_8bit=config.use_quantization) if config.device == "cuda" else None
            self._model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                quantization_config=quantization_config,
                device_map="auto" if config.device == "cuda" else None,
                torch_dtype=getattr(torch, getattr(config, "torch_dtype", "float32"))
            ).eval()
            if config.use_torch_compile and config.device == "cuda":
                self._model = torch.compile(self._model)
            logger.info(f"Loaded empathetic LLM model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            self._tokenizer = None

    def generate_empathetic_response(self, input_text: str, history: List = None, emotions: Dict = None, verbosity: int = 2) -> str:
        if history is None:
            history = []
        if emotions is None:
            emotions = {}
        dominant_emotion = max(emotions, key=emotions.get, default="neutral")
        
        try:
            self._ensure_model()
            if self._model and self._tokenizer:
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
                prompt = f"""
You are EmoDude, a compassionate AI therapist specializing in emotional support, CBT, DBT, and person-centered therapy.
User's post: {input_text}

Detected emotions: {emotions}

Provide an empathetic response that:
1. Starts with deep, heartfelt validation of the user's emotions (100+ words)
2. Acknowledges their efforts and love behind their actions
3. Normalizes their feelings as human and valid
4. Offers a gentle, open-ended invitation to share more (always end with a question)
5. Avoids generic phrases like "Let's dive into what brought you here"
Use warm, professional, and deeply compassionate language. Length: {'short' if verbosity==1 else 'balanced (150+ words)' if verbosity==2 else 'detailed (250+ words)'}.

Make it personal and engaging, drawing from the user's specific story.
                """
                inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                max_len = 150 if verbosity == 1 else 250 if verbosity == 2 else 400
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_length=max_len,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id
                    )
                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True).split(prompt)[-1].strip()
                if len(response) < 100 or '?' not in response:  # Ensure length and question
                    response = self._get_template_response(dominant_emotion, verbosity)
                logger.info(f"Generated empathetic response")
                return response
            else:
                return self._get_template_response(dominant_emotion, verbosity)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_template_response(dominant_emotion, verbosity)

    def _get_template_response(self, emotion: str, verbosity: int) -> str:
        templates = self.therapeutic_templates.get(emotion, self.therapeutic_templates["neutral"])
        response = random.choice(templates)
        if verbosity == 1:
            return response.split('.')[0] + "."
        elif verbosity == 3:
            extensions = {
                "sadness": " Your courage in sharing this is a step toward healing. Let’s find a small way to honor your feelings. What memory would you like to revisit?",
                "grief": " Your love is so evident in this pain. Perhaps we can find a way to keep their memory alive together. What ritual feels right?",
                "anger": " Your passion shows how much this matters. Let’s explore a way to channel this energy. What boundary needs strengthening?",
                "remorse": " Self-forgiveness is a journey, and you're not alone. What's one kind thing you can do for yourself today?",
                "guilt": " Your heart's in the right place. Let's think of one small step toward self-compassion. What would forgiveness sound like?",
                "frustration": " This intensity shows your strength. What's one thing we can do to ease this storm? How can we pivot creatively?",
                "helplessness": " Even small steps can light the way. What's one thing that feels within reach today?",
                "love": " Your love is a gift. How can we celebrate it today? What nourishes this feeling?",
                "admiration": " Your awe reflects your open heart. Let's explore how this can inspire your next steps. What quality will you embody?",
                "fear": " Fear is a messenger. What wisdom does it bring, and how can we thank it while moving forward? What's one safe step?",
                "joy": " Joy is your natural state. How can we invite more of it into your life? What amplifies this delight?",
                "disappointment": " Disappointment carves space for new growth. What seed can we plant in this space? What's the lesson here?",
                "confusion": " Confusion is the prelude to clarity. Trust the unfolding—what emerges when you pause? What's one question to ask?",
                "excitement": " Excitement is momentum. Ride this wave—what destination calls to you? How can we prepare?",
                "gratitude": " Gratitude multiplies abundance. What ripples does this thanks create? Who else can you share it with?",
                "pride": " Pride is earned fuel. How will you use this energy to propel forward? What's the next milestone?",
                "relief": " Relief is permission to rest. Savor it fully—what recharges you now? How can we extend this calm?",
                "surprise": " Surprise shakes the ordinary into magic. What new perspective opens here? What's the gift in this twist?",
                "self_doubt": " Self-doubt is a shadow; your light is brighter. What truth dispels it? What's one strength to claim?",
                "hope": " Hope is the bridge from here to there. Walk it one step at a time. What possibility excites you most?",
                "courage": " Courage grows in the doing. You've already begun—what's the next brave act? What support do you need?",
                "neutral": " Neutrality is fertile ground. What will you plant in this spaciousness? What's arising in the quiet?"
            }
            return response + extensions.get(emotion, " Let's explore what feels right for you next. What comes up for you?")
        return response

    def generate_therapeutic_solutions(self, input_text: str, emotions: Dict[str, float]) -> Dict[str, str]:
        try:
            dominant_emotion = max(emotions, key=emotions.get, default="neutral")
            problems = self._identify_problems(input_text, dominant_emotion)
            solutions = {}
            for problem in problems:
                strategy_key = self._map_problem_to_strategy(problem)
                solutions[problem] = random.choice(self.solution_strategies.get(strategy_key, ["Consider therapy for personalized support. Explore resources like Psychology Today for local professionals."]))
            return solutions
        except Exception as e:
            logger.error(f"Error generating solutions: {e}")
            return {"General Support": "Consider speaking with a therapist for personalized guidance. Start with self-care: deep breaths, a walk in nature, or journaling your feelings."}

    def _identify_problems(self, text: str, emotion: str) -> List[str]:
        text_lower = text.lower()
        problems = []
        if any(word in text_lower for word in ['lost', 'died', 'dead', 'gone', 'grief', 'mourning', 'sad', 'heartbroken', 'recipe', 'butterfly']):
            problems.append("Grief and Loss")
        if any(word in text_lower for word in ['guilty', 'fault', 'blame', 'shame', 'should have', 'failed']):
            problems.append("Guilt and Regret")
        if any(word in text_lower for word in ['neighbor', 'hurt', 'betrayed', 'argument', 'conflict']):
            problems.append("Interpersonal Conflict")
        if emotion in ['anger', 'annoyance', 'frustration']:
            problems.append("Anger Management")
        if emotion in ['helplessness', 'powerlessness']:
            problems.append("Helplessness")
        if emotion in ['love', 'caring', 'joy', 'attachment']:
            problems.append("Nurturing Love and Connection")
        if emotion in ['self_doubt', 'insecurity']:
            problems.append("Self-Doubt")
        if emotion in ['fear', 'anxiety']:
            problems.append("Fear and Anxiety")
        if emotion in ['hope', 'optimism']:
            problems.append("Building Hope")
        if emotion in ['courage', 'bravery']:
            problems.append("Cultivating Courage")
        return problems if problems else [f"{emotion.title()} Exploration"]

    def _map_problem_to_strategy(self, problem: str) -> str:
        mapping = {
            "Grief and Loss": "grief_loss",
            "Guilt and Regret": "guilt_regret",
            "Interpersonal Conflict": "anger_management",  # Overlap
            "Anger Management": "anger_management",
            "Helplessness": "helplessness",
            "Nurturing Love and Connection": "nurturing_love_and_connection",
            "Self-Doubt": "self_doubt",
            "Fear and Anxiety": "fear_anxiety",
            "Building Hope": "hope_optimism",
            "Cultivating Courage": "courage_bravery"
        }
        return mapping.get(problem, "general_support")

response_generator = ResponseGenerator()