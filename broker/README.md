# About broker/

## Structure
```
├── base_interface.py       # Interface that the main broker classes implements/uses
├── README.md               # Documentation related to broker/
├── capital_com/            # Capital.com API integration
│   └── capital_com.py/     # Main orchestrator file used to interact with different endpoints of API
│       ├── rest_api/       # REST API functionality. Each class file follows the endpoints from official doc
│       │   └─ ...
│       └── web_socket/     # Web Socket functionality (streaming)
└── ...
```

## Useful infor about the different brokers
### Capital.com (API)
Capital.com is currently the only supported broker.<br>
The reason for this choice is the large amount of symbols to pick from, low costs, good UI, and okay API and documentation.
#### Useful links:
- [API documentation](https://open-api.capital.com/)
- [Requirements to use their API](https://open-api.capital.com/#section/Getting-started)