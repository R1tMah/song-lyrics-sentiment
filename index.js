const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 3001;

app.use(bodyParser.json());

app.post('/analyze', async (req, res) => {
    try {
        const data = req.body.text; // Assume the text data is sent in the request body

        // Call the Python script using an external API (could be localhost)
        const response = await axios.post('http://localhost:3000/analyze', { text: data });

        // Send the response back to the client
        res.json(response.data);
    } catch (error) {
        console.error(error);
        res.status(500).send('Error processing the request');
    }
});


app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
