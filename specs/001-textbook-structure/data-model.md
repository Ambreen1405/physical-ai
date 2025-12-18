# Data Model: Physical AI & Humanoid Robotics Textbook

## Entities

### Chapter
**Description**: A structured learning unit containing educational content

**Fields**:
- id: string (unique identifier for the chapter)
- title: string (display title of the chapter)
- sidebar_position: number (position in sidebar navigation)
- description: string (SEO description for the chapter)
- keywords: string[] (SEO keywords for search optimization)
- learning_objectives: string[] (list of learning objectives for the chapter)
- content: string (main content in Markdown/MDX format)
- code_examples: CodeExample[] (collection of code examples in the chapter)
- practical_exercises: PracticalExercise[] (step-by-step exercises)
- assessment_questions: AssessmentQuestion[] (questions for assessment)
- further_reading: Reference[] (IEEE format references)

**Validation Rules**:
- id must be unique across all chapters
- title must not be empty
- sidebar_position must be a positive number
- learning_objectives must contain at least one objective
- content must follow Markdown/MDX format

**State Transitions**: N/A (static content entity)

### Module
**Description**: A collection of related chapters organized by topic

**Fields**:
- id: string (unique identifier for the module)
- title: string (display title of the module)
- description: string (brief description of the module)
- chapters: Chapter[] (collection of chapters in the module)
- assessment: Chapter (assessment chapter for the module)
- order: number (sequence order of the module in the textbook)

**Validation Rules**:
- id must be unique across all modules
- title must not be empty
- must contain at least one chapter
- order must be a positive number

**State Transitions**: N/A (static content entity)

### CodeExample
**Description**: A code snippet with syntax highlighting and explanation

**Fields**:
- id: string (unique identifier for the code example)
- language: string (programming language, e.g., "python", "bash")
- code: string (the actual code content)
- explanation: string (explanation of what the code does)
- file_path: string (path where the code should be saved, if applicable)
- line_numbers: boolean (whether to show line numbers)

**Validation Rules**:
- id must be unique within the parent chapter
- language must be a supported syntax highlighting language
- code must not be empty
- language must be valid (python, bash, c++, etc.)

**State Transitions**: N/A (static content entity)

### PracticalExercise
**Description**: A step-by-step tutorial or hands-on activity

**Fields**:
- id: string (unique identifier for the exercise)
- title: string (title of the exercise)
- steps: string[] (ordered list of steps to complete the exercise)
- prerequisites: string[] (requirements to complete the exercise)
- expected_outcome: string (what the user should achieve)
- difficulty: "beginner" | "intermediate" | "advanced" (difficulty level)

**Validation Rules**:
- id must be unique within the parent chapter
- title must not be empty
- steps must contain at least one step
- difficulty must be one of the allowed values

**State Transitions**: N/A (static content entity)

### AssessmentQuestion
**Description**: A question used for assessment and evaluation

**Fields**:
- id: string (unique identifier for the question)
- question: string (the actual question text)
- options?: string[] (multiple choice options, if applicable)
- correct_answer: string (the correct answer)
- explanation: string (explanation of the correct answer)
- difficulty: "beginner" | "intermediate" | "advanced" (difficulty level)
- question_type: "multiple-choice" | "short-answer" | "essay" | "code" (type of question)

**Validation Rules**:
- id must be unique within the parent chapter
- question must not be empty
- correct_answer must not be empty
- difficulty must be one of the allowed values
- question_type must be one of the specified types
- if question_type is "multiple-choice", options must be provided

**State Transitions**: N/A (static content entity)

### Reference
**Description**: An academic reference in IEEE format

**Fields**:
- id: string (unique identifier for the reference)
- title: string (title of the referenced work)
- authors: string[] (list of authors)
- publication: string (journal, conference, or book title)
- year: number (publication year)
- url?: string (optional URL to the reference)
- doi?: string (optional DOI identifier)

**Validation Rules**:
- id must be unique within the parent chapter
- title must not be empty
- authors must contain at least one author
- year must be a valid year (not in the future)
- if DOI is provided, it must follow proper DOI format

**State Transitions**: N/A (static content entity)

### Student
**Description**: The primary user of the textbook (conceptual entity for user stories)

**Fields**:
- id: string (unique identifier for the student)
- name: string (display name of the student)
- progress: ModuleProgress[] (tracking progress through modules)
- preferences: UserPreferences (user-specific preferences)

**Validation Rules**:
- id must be unique across all students
- name must not be empty

**State Transitions**: N/A (this is a conceptual entity, actual student data would be handled by external systems)

### ModuleProgress
**Description**: Tracks a student's progress through a specific module

**Fields**:
- module_id: string (reference to the module)
- completed_chapters: string[] (list of completed chapter IDs)
- current_chapter: string (currently active chapter ID)
- assessment_score?: number (score on module assessment, if completed)
- completion_date?: string (date when module was completed)

**Validation Rules**:
- module_id must reference an existing module
- completed_chapters must be valid chapter IDs within the module
- current_chapter must be a valid chapter ID within the module
- assessment_score must be between 0 and 100 if provided

### UserPreferences
**Description**: User-specific preferences for the textbook experience

**Fields**:
- theme: "light" | "dark" (current theme preference)
- font_size: "small" | "medium" | "large" (font size preference)
- code_theme: string (preferred code syntax highlighting theme)
- accessibility_settings: AccessibilitySettings (accessibility preferences)

### AccessibilitySettings
**Description**: Accessibility-related preferences

**Fields**:
- high_contrast: boolean (whether to use high contrast mode)
- screen_reader_friendly: boolean (optimize for screen readers)
- reduced_motion: boolean (reduce animations)
- keyboard_navigation: boolean (enhanced keyboard navigation)

## Relationships

- Module contains many Chapters
- Chapter contains many CodeExamples
- Chapter contains many PracticalExercises
- Chapter contains many AssessmentQuestions
- Chapter contains many References
- Student has many ModuleProgress records
- ModuleProgress references a Module
- Student has one UserPreferences
- UserPreferences has one AccessibilitySettings