def test_service_is_denied_prediction(client):
    response = client.post(
        "/predict",
        json={"text": "Cardiovascular pain."},
        headers={"X-API-Key": "srv-" + "0"*30}
    )
    assert response.status_code == 403
    assert response.json()["error_code"] == "forbidden"