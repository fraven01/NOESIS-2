# RAG Tools: User Integration Concept ("Erlebnis-Workbench")

**Objective**: Make the new "User Integration" features (Attribution, Permissions, Activity, Collaboration) tangible for developers within the `rag-tools` workbench.

## Core Concept: "Identity Simulation"

To test multi-user scenarios (permissions, notifications, sharing) without logging in/out constantly, we introduce an **Identity Simulator** in the workbench.

### 1. Identity Switcher (Header Control)

- **UI**: A dropdown in the `rag-tools` header: "Acting As: [ Current User ]".
- **Options**:
  - **Me** (Real logged-in user)
  - **Simulated Users** (e.g., "Alice (Admin)", "Bob (Legal)", "Charlie (External)") -> *Requires creating fixture users*.
  - **Anonymous** (Unauthenticated simulation)
- **Mechanism**:
  - Sets `request.session['rag_tools_simulated_user_id']`.
  - Middleware or View Decorator overrides `request.user` or `scope.user_id` for `rag-tools` views based on this session key.

## Functional Modules

### 2. Document Explorer Enhancements

**Goal**: Visualize ownership and permissions.

- **Columns**: Add `Created By` (Avatar/Name) and `Access Level` (Owner/Read/Write).
- **Actions**:
  - **" Inspect Permissions"**: Opens a modal showing the `DocumentAuthzService` decision tree for the *current simulated user* vs. the document.
    - *Visual*: "Allowed by Case Membership" or "Allowed by Explicit Grant".
  - **"View Activity"**: Shows the `DocumentActivity` log for this document.

### 3. Collaboration Playground (New Tab)

**Goal**: Test interaction loops.

- **Target**: Select a Document ID.
- **Tools**:
  - **Comments**: Post a comment as the simulated user. Support `@mentions` (autocomplete simulated users).
  - **Favorites**: Toggle favorite status.
  - **Share**: Grant permission to another user.
- **Feedback**:
  - **"Incoming Notifications"**: A real-time (polling or HTMX refresh) list of notifications received by the *simulated user*.
  - **"External Events"**: Log of triggered `NotificationEvent`s (simulating email dispatch).

### 4. Search & Access Verification

**Goal**: Verify visibility rules.

- **Search Tab Update**:
  - Results now visually indicate *why* a document is visible (e.g., "Matched via Case: Legal").
  - "Shadow Results": Toggle to show documents that *matched the query* but were *hidden by permissions* (greyed out), to debug authorization.

## Implementation Steps (Draft)

1. **Fixtures**: Create a standard set of dev users (Admin, Legal, Regular, External).
2. **Middleware/Decorator**: `SimulateUserMiddleware` for `rag-tools/*` routes.
3. **UI Updates**:
    - Header dropdown.
    - New columns in Document Explorer.
    - New "Collaboration" tab.
    - Updates to `tool_search` view to expose authz rationale.

## User Story Example

1. Developer selects "Alice (Admin)".
2. Uploads `secret_strategy.pdf` via **Ingestion Tab**.
3. Switches to "Bob (External)".
4. Tries to **Search** for "strategy" -> No results.
5. Switches to "Alice".
6. **Shares** `secret_strategy.pdf` with "Bob".
7. Switches to "Bob".
8. **Notifications**: "Alice shared a document...".
9. **Search**: "strategy" -> Result appears.
10. **Comment**: "Thanks Alice!" -> posted as Bob.
