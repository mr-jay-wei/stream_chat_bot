graph TD
    A[Document] --> B[HTML];
    B --> C[HEAD];
    B --> D[BODY];

    C --> C1[TITLE: "企业级..."];
    C --> C2[META];
    C --> C3[LINK];

    D --> D1[DIV class="container"];
    D1 --> D1_1[H1: "🤖 ..."];
    D1 --> D1_2[DIV id="connectionStatus"];
    D1 --> D1_3[DIV id="chatContainer"];
    D1 --> D1_4[DIV class="input-container"];

    D1_3 --> D1_3_1[DIV class="message status-message"];
    D1_4 --> D1_4_1[INPUT id="questionInput"];
    D1_4 --> D1_4_2[BUTTON id="sendButton"];