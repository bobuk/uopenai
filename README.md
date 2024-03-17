# uOpenAI

uOpenAI is a lightweight, asynchronous Python library designed to interact with the OpenAI API. Leveraging the power of `httpx` for async HTTP requests and `pydantic` for data validation and settings management, it aims to provide a seamless and efficient way to access OpenAI's capabilities in modern async Python applications.

## Features

-   **Fully Asynchronous**: Built from the ground up for asyncio compatibility.
-   **Type-Hinted**: Ensures robust code through comprehensive type annotations.
-   **Simple and Intuitive**: A straightforward API to access OpenAI's features with minimal boilerplate.

## Installation

To install uOpenAI, run:

```bash
pip install uopenai
```

## Quick Start

Here's a quick example to get you started:

```python
import asyncio
from uopenai import OpenAI, Message

async def main():
    api = OpenAI("your-api-key")
    # Complete a prompt with options
    res = await api.complete([Message("system", "You are a chat bot"), Message("user", "What’s on your mind?")], max_tokens=50)
    print(res.choices[0].message.content)

    # easy version

    res = await api.easy_complete("What’s on your mind?", "You are a chat bot")

    res = await api.easy_complete("What’s on your mind?",
                    "You are a JSON answering machine, give the answer in JSON only.", json=True)

if __name__ == "__main__":
    asyncio.run(main())
```

Replace "your-api-key" with your actual OpenAI API key.

## Requirements

-   Python 3.7+
-   OpenAI API key
-   `httpx`
-   `pydantic`

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

uOpenAI is released under the The Unlicense. See the LICENSE file for more details.
