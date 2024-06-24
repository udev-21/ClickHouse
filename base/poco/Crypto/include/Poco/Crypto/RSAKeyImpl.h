//
// RSAKeyImpl.h
//
// Library: Crypto
// Package: RSA
// Module:  RSAKeyImpl
//
// Definition of the RSAKeyImpl class.
//
// Copyright (c) 2008, Applied Informatics Software Engineering GmbH.
// and Contributors.
//
// SPDX-License-Identifier:	BSL-1.0
//


#ifndef Crypto_RSAKeyImplImpl_INCLUDED
#define Crypto_RSAKeyImplImpl_INCLUDED


#include <istream>
#include <ostream>
#include <vector>
#include <wolfssl/options.h>

#include "Poco/AutoPtr.h"
#include "Poco/Crypto/Crypto.h"
#include "Poco/Crypto/EVPPKey.h"
#include "Poco/Crypto/KeyPairImpl.h"
#include "Poco/Crypto/OpenSSLInitializer.h"
#include "Poco/RefCountedObject.h"


#include <openssl/compat_types.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>
#include <openssl/bn.h>

struct bignum_st;
struct rsa_st;
// typedef struct bignum_st BIGNUM;
// typedef struct rsa_st RSA;


namespace Poco
{
namespace Crypto
{


    class X509Certificate;
    class PKCS12Container;


    class RSAKeyImpl : public KeyPairImpl
    /// class RSAKeyImpl
    {
    public:
        typedef Poco::AutoPtr<RSAKeyImpl> Ptr;
        typedef std::vector<unsigned char> ByteVec;

        RSAKeyImpl(const EVPPKey & key);
        /// Constructs ECKeyImpl by extracting the EC key.

        RSAKeyImpl(const X509Certificate & cert);
        /// Extracts the RSA public key from the given certificate.

        RSAKeyImpl(const PKCS12Container & cert);
        /// Extracts the EC private key from the given certificate.

        RSAKeyImpl(int keyLength, unsigned long exponent);
        /// Creates the RSAKey. Creates a new public/private keypair using the given parameters.
        /// Can be used to sign data and verify signatures.

        RSAKeyImpl(const std::string & publicKeyFile, const std::string & privateKeyFile, const std::string & privateKeyPassphrase);
        /// Creates the RSAKey, by reading public and private key from the given files and
        /// using the given passphrase for the private key. Can only by used for signing if
        /// a private key is available.

        RSAKeyImpl(std::istream * pPublicKeyStream, std::istream * pPrivateKeyStream, const std::string & privateKeyPassphrase);
        /// Creates the RSAKey. Can only by used for signing if pPrivKey
        /// is not null. If a private key file is specified, you don't need to
        /// specify a public key file. OpenSSL will auto-create it from the private key.

        ~RSAKeyImpl();
        /// Destroys the RSAKeyImpl.

        RSA * getRSA();
        /// Returns the OpenSSL RSA object.

        const RSA * getRSA() const;
        /// Returns the OpenSSL RSA object.

        int size() const;
        /// Returns the RSA modulus size.

        ByteVec modulus() const;
        /// Returns the RSA modulus.

        ByteVec encryptionExponent() const;
        /// Returns the RSA encryption exponent.

        ByteVec decryptionExponent() const;
        /// Returns the RSA decryption exponent.

    private:
        RSAKeyImpl();

        void freeRSA();
        static ByteVec convertToByteVec(const BIGNUM * bn);

        RSA * _pRSA;
    };


    //
    // inlines
    //
    inline RSA * RSAKeyImpl::getRSA()
    {
        return _pRSA;
    }


    inline const RSA * RSAKeyImpl::getRSA() const
    {
        return _pRSA;
    }


}
} // namespace Poco::Crypto


#endif // Crypto_RSAKeyImplImpl_INCLUDED
