RS_PROMPTS = {
    "PHASE0_RS_INIT": """Based on Q: '{Q}' and the image, identify the most suitable referring phrases for the target.
    Guideline:
    - Adaptive Specificity: 
      1. If the instance is unique/obvious, use a generic name (e.g., 'dog', 'sky').
      2. If there are multiple similar instances, MUST use attributes/location to distinguish (e.g., 'black dog', 'dog on left').

    STRICT VISUAL CONSTRAINTS:
    1. The target MUST be currently visible and occupy actual pixels in this image.
    2. Do NOT output objects that are merely implied by the text logic. Only name what is physically seen.
    
    Output ONLY a comma-separated list:""",

    "PHASE0_ITERATIVE_REFINE": """Context: User Question Q: '{Q}'.
    Current Best Candidate: '{current_best}'.
    
    Task: Check if there is a **more specific** referring phrase for the target in the image.
    Instruction:
    - If '{current_best}' is already good: Output ONLY the word 'NO'.
    - If you can improve it (e.g., add color/location): Output **ONLY the new phrase**. 
    
    Output:""",

    "REFINE_TEMPLATE": """Context: User Question Q: '{Q}'. 
    Target Object: '{T_final}'.
    Task: Look at the image. Generate 3 distinct noun phrases that refer to EXACTLY THE SAME INSTANCE as '{T_final}'.
    CRITICAL CONSTRAINT:
    - The new phrase must still be a valid answer to Q.
    - Length: 2 to 5 words.
    Output 3 phrases separated by commas:""",

    "REFINE_SIMPLIFY": """Context: User Question Q: '{Q}'. Target: '{T_final}'.
    Current Phrase: '{P_old}'.
    Task: Refine '{P_old}' to precisely describe the **specific instance** implied by the Question Q.
    Output ONLY the new phrase:""",

    "SEMANTIC_GATE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: The same image with a **{color_name} mask overlay**.

    Context:
    - **User Query (Ground Truth)**: '{T_final}' (derived from Q: '{Q}')

    Task: STRICTLY Verify if the **{color_name} mask** matches the User Query.

    Logic:
    - The mask MUST satisfy ALL adjectives in the query.
    - If the mask covers the background or the WRONG object, output 0.0.

    Output ONLY: 1.0 (Pass) or 0.0 (Fail).""",

    "ARENA_JUDGE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: Candidate A (Overlay with {color_name}).
    - **Image 3**: Candidate B (Overlay with {color_name}).

    Context: 
    - **User Query (Ground Truth)**: '{Q}'
    - **Candidate A Source**: '{P_A}'
    - **Candidate B Source**: '{P_B}'

    Task: Judge which mask better satisfies the **User Query (Q)**.

    **PRIORITY #1: SEMANTIC CORRECTNESS (THE GOLDEN RULE)**
    - Does the mask cover the **exact specific object** described in Q?
    - **CRITICAL**: If Candidate A covers the WRONG object (e.g., wrong color, wrong position, wrong type) and Candidate B covers the RIGHT object, **Candidate B WINS IMMEDIATELY**, regardless of how messy it is.

    **PRIORITY #2: SEGMENTATION QUALITY (Only if both are semantically correct)**
    -If (and ONLY if) both masks cover the correct object, choose the one with tighter boundaries, better completeness, and less background noise.
    
    Output ONLY: 'A' (if Image 2 is better) or 'B' (if Image 3 is better).""",

    "PHASE_PRE_BOX_DETECT": """Context: User Question Q: '{Q}'. 
    Task: Detect the instance described in Question Q: '{Q}'. 

    Instruction:
    1. Identify the best answer in the image based on question Q.
    2. Output a TIGHT bounding box that strictly encloses ONLY the instance.
    3. Use ABSOLUTE PIXEL COORDINATES based on the image size {width}x{height}.

    Format: [x1, y1, x2, y2]
    Output ONLY the list [x1, y1, x2, y2]:""",

    "DIRECT_SCORE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: The same image with a **{color_name} mask overlay**.

    Context:
    - **User Query**: '{Q}'
    - **Candidate Phrase**: '{P}'

    Task: Rate how well the mask matches the User Query on a scale from 0 to 100.
    
    Criteria:
    - 0-20: Completely wrong object or background.
    - 21-50: Partially correct but covers significant wrong areas or misses parts.
    - 51-80: Correct object but boundaries are loose or slightly inaccurate.
    - 81-100: Perfect match, tight boundaries, semantically accurate.

    Output ONLY the number (0-100):"""
}

RES_PROMPTS = {
    "PHASE0_RS_INIT": """Context: User Referring Expression Q: '{Q}'.
    
    Task: Identify the core target object implied by Q and output it as a simple noun phrase.
    
    Format: Output the phrase inside curly braces.
    Example: {{red apple}}
    
    Output:""",

    "PHASE0_RES_NORMALIZATION": """Context: User Referring Expression Q: '{Q}'.
    
    Task: Rewrite Q into a grammatically complete, subject-first noun phrase.
    
    Requirements:
    1. Identify the main subject (e.g., "guy", "cat", "car").
    2. Convert "attribute subject" (e.g., "red shirt man") into "subject with attribute" (e.g., "the man in the red shirt") if it sounds more natural, OR keep it if it is already clear.
    3. CRITICAL: Do NOT lose any semantic meaning (color, position, relation) from the original Q.
    4. Start with "The" or "A".
    
    Format Constraints:
    - Do NOT output any explanation or reasoning.
    - Do NOT simplify specific objects into generic ones (e.g., do NOT change 'blue car' to 'parked vehicle').
    - Output the final phrase enclosed in curly braces {{...}}.
    
    Example:
    Input: "hat bench guy"
    Output: {{The guy with a hat on the bench}}
    
    Input: "left bottom pizza"
    Output: {{The pizza on the bottom left}}
    
    Output:""",

    "REFINE_SIMPLIFY": """Context: User Query Q: '{Q}'.
    Current Phrase: '{P_old}'.

    Task: Refine '{P_old}' to be a short, SAM-friendly phrase.
    
    AVOID AMBIGUITY:
    - Keep discriminative tokens that ensure uniqueness in the scene:
      color, spatial (left/right/top/bottom), numerals, and relations.
    - Convert spatial or attributive comparatives to SUPERLATIVES to strictly define the instance.
       - "person on the right" -> "rightmost person"
       - "upper bird" -> "topmost bird"
       - "larger bowl" -> "largest bowl"
    
    Output the result inside curly braces {{...}}:""",

    "REFINE_TEMPLATE": """Context: User Query Q: '{Q}'.
    Target Object: '{T_final}'.
    
    Task: Generate 3 distinct noun phrases by {T_final} that refer to **EXACTLY THE SAME INSTANCE** as {Q}.
        - Convert spatial or attributive comparatives to SUPERLATIVES to strictly define the instance.
        - "person on the right" -> "rightmost person"
        - "upper bird" -> "topmost bird"
        - "larger bowl" -> "largest bowl"
    
    Format: Output 3 phrases separated by commas, NO brackets, just text.
    Output:""",

    "PHASE_PRE_BOX_DETECT": """Context: User Referring Expression Q: '{Q}'.
    Image size: width={width}, height={height}.
    
    Task: Detect the specific target instance described by {Q} and output a SINGLE, TIGHT bounding box.
    Format: [x1, y1, x2, y2]
    Output ONLY the list [x1, y1, x2, y2]:""",

    "PHASE0_ITERATIVE_REFINE": """Context: User Query Q: '{Q}'.
    Current Candidate: '{current_best}'.
    
    Task: Ensure '{current_best}' refers to the EXACT SAME physical instance as Q.
    
    Instruction:
    - If it lost any discriminative token (color/position/relation), output the refined phrase inside curly braces {{...}}.
    - If it is already good and precise, output {{NO}}.
    
    Output:""",

    "SEMANTIC_GATE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: The same image with a **{color_name} mask overlay**.

    Context:
    - **User Query (Ground Truth)**: '{Q}'

    Task: STRICTLY Verify if the **{color_name} mask** matches the User Query.

    CRITICAL FAILURE CHECKS (If any is true, Output 0.0):
    1. **Color Mismatch**: Query says "blue" but the masked object is red/silver/white? -> FAIL.
    2. **Position Mismatch**: Query says "left" but mask is in the middle/right? -> FAIL.
    3. **Count Mismatch**: Query implies ONE object but mask covers multiple? -> FAIL.
    4. **Wrong Object**: Query says "car" but mask is "bus"? -> FAIL.

    Logic:
    - The mask MUST satisfy ALL adjectives in Q.
    - A "perfect looking mask" on the WRONG object is 0.0.

    Output ONLY: 1.0 (Pass) or 0.0 (Fail).""",

    "ARENA_JUDGE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: Candidate A (Overlay with {color_name}).
    - **Image 3**: Candidate B (Overlay with {color_name}).

    Context: 
    - **User Query (Ground Truth)**: '{Q}'
    - **Candidate A Source**: '{P_A}'
    - **Candidate B Source**: '{P_B}'

    Task: Judge which mask better satisfies the **User Query (Q)**.

    **PRIORITY #1: SEMANTIC CORRECTNESS (THE GOLDEN RULE)**
    - Does the mask cover the **exact specific object** described in Q?
    - **CRITICAL**: If Candidate A covers the WRONG object (e.g., wrong color, wrong position, wrong type) and Candidate B covers the RIGHT object, **Candidate B WINS IMMEDIATELY**, regardless of how messy it is.
    - **Specificity**: If Q is "the man on the far left", and Candidate A captures strictly the leftmost man, while Candidate B captures a group or the wrong man, Candidate A WINS.

    **PRIORITY #2: SEGMENTATION QUALITY (Only if both are semantically correct)**
    -If Q implies the object is leaving the frame/cropped, do NOT penalize Mask for being incomplete or touching the edge. 
    -If (and ONLY if) both masks cover the correct object, choose the one with tighter boundaries, better completeness, and less background noise.
    
    Output ONLY: 'A' (if Image 2 is better) or 'B' (if Image 3 is better).""",

    # --- Common / Ablation ---
    "DIRECT_SCORE": """Input Provided:
    - **Image 1**: The original raw image.
    - **Image 2**: The same image with a **{color_name} mask overlay**.

    Context:
    - **User Query**: '{Q}'
    - **Candidate Phrase**: '{P}'

    Task: Rate how well the mask matches the User Query on a scale from 0 to 100.
    
    Criteria:
    - 0-20: Completely wrong object or background.
    - 21-50: Partially correct but covers significant wrong areas or misses parts.
    - 51-80: Correct object but boundaries are loose or slightly inaccurate.
    - 81-100: Perfect match, tight boundaries, semantically accurate.

    Output ONLY the number (0-100):"""
}

# Default to RS if accessed directly 
MLLM_PROMPT_DEFS = RS_PROMPTS
